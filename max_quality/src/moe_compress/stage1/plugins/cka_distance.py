"""Per-layer CKA pairwise distance matrices for downstream GRAPE merge.

Papers
------
This plugin sits at the intersection of two papers:

1. **Kornblith et al.**, "Similarity of Neural Network Representations
   Revisited", ICML 2019 — arXiv:1905.00414. Defines the CKA
   similarity primitive: ``CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) *
   HSIC(Y, Y))`` with the biased HSIC linear-kernel centering of Gretton
   (2005): ``K_c = K - row_mean - col_mean + grand_mean``. **Faithfully
   implemented here** for both the GPU-batched and CPU-per-pair paths.

   Official reference implementation: google-research/google-research
   ``representation_similarity/Demo.ipynb`` @ directory-touching commit
   ``89e3921863e276cdbe49bd25077905f75e981f4e`` (2019-06-10) —
   github.com/google-research/google-research/blob/89e3921863e276cdbe49bd25077905f75e981f4e/representation_similarity/Demo.ipynb.

2. **Zhang et al.** (GRAPE), "Does a Global Perspective Help Prune
   Sparse MoEs Elegantly?" — arXiv:2604.06542. §3.2 defines
   ``D^l ∈ ℝ^{N×N}`` as a *similarity* matrix
   (source.md L245-L249: "Let D^l ... denote the pairwise similarity
   matrix of experts in the l-th MoE layer... D^l can be instantiated
   using CKA (Davari et al.), mean squared error, or other similarity
   measures."). Algorithm 1 lines 8-9 select the most-redundant layer
   with ``argmax R^l`` (where ``R^l = Σ_{i≠j} D^l_{ij}``) and the
   most-similar within-layer pair with ``argmax D^l``.

   No official code repository for GRAPE was published at paper release
   (verified 2026-05; arxiv.org/abs/2604.06542 has no code link, the
   first author's GitHub has no GRAPE repo). This stage's GRAPE
   greedy-merge implementation is therefore reference-free.

Deviation: D-cka-distance
-------------------------
The plugin builds ``D^l_{ij} = 1 − CKA(f_i, f_j)`` (distance form,
``[0, 1]``: 0 = identical, 1 = maximally different), inverting the GRAPE
paper's similarity-form ``D^l``. Downstream consumers (the GRAPE merge
plugin) use ``argmin`` on the distance matrix instead of the paper's
``argmax`` on the similarity matrix; they also evaluate the redundancy
score ``R^l = Σ_{i≠j} D^l_{ij}`` as a *distance* sum (smaller = more
redundant) instead of the paper's *similarity* sum (larger = more
redundant) and use ``argmin R^l`` accordingly.

**The transformation is a sign-flip, not a numerical deviation.** Under
the identity ``R^l_distance = N(N−1) − R^l_similarity`` (constant offset,
which holds because the ``i = j`` diagonal is excluded; the diagonal
would break the offset since ``1 − CKA(f_i, f_i) = 0`` vs
``CKA(f_i, f_i) = 1``), layer ranking, pair ranking, and the final
budget allocation are mathematically equivalent. Polarity of the
Eq. (3) normalization is inverted in the distance form (``R̃^l_dist =
1 − R̃^l_sim``) but the cross-layer rank order is preserved.

Why the distance form: the merge plugin's redundancy criterion reads as
"higher = more redundant by Σ-of-distances" with the standard
small-distance / near-duplicate intuition; the ``argmin`` flow matches
how Phase F's greedy queue is consumed elsewhere in the stage.

Output context contract
-----------------------
- ``reads``: ``output_acc`` (the calibration-pass-populated
  ``ExpertOutputAccumulator`` — per-(layer, expert) output reservoir,
  cap 256 tokens/expert), ``moe_layers``, ``config``.
- ``writes``: ``D_matrices`` — ``dict[int, torch.Tensor]`` mapping
  ``layer_idx`` to the ``N×N`` distance matrix (CPU fp32).
- ``provides``: ``("output_reservoir",)`` — declarative metadata
  advertising the per-(layer, expert) output reservoir as a hook needed
  during the calibration pass.

``contribute_artifact`` returns ``{}`` — ``D_matrices`` is consumed
in-memory by the downstream merge plugin and is never written to disk.

Implementation paths
--------------------
Two byte-equivalent (within fp32 tolerance) paths share the same biased
HSIC centering:

- **GPU vectorized** (default for prod): subsamples every active expert
  to a single ``m_min`` over the active set, batches the Gram matrices,
  computes the full ``N×N`` HSIC table in O(N · m² · d) GPU work.
  ~1 sec/layer on H200 vs ~10 min/layer for the CPU per-pair path.
- **CPU per-pair fallback**: original implementation. Activated when
  the vectorized path is unsafe — ``m_min < 32`` OR ``m_min < m_max // 4``.
  Used by tests with tiny calibration sets and as a safety net when
  reservoir under-fill would force every pair to use a low ``m``.

With the prod default of ``num_calibration_samples = 1024`` and the
``ExpertOutputAccumulator`` reservoir cap of 256 tokens/expert, all
active experts saturate at ``m = 256`` and the GPU path is bit-equivalent
(within fp32 tolerance) to the original.

A weight-space ablation override (``cosine`` / ``mse`` via
``stage1_grape.similarity_metric``) is supplied for cross-metric
sanity tests; it computes distances on concatenated
``gate_proj | up_proj | down_proj`` weight banks and produces an
incommensurable ``R`` scale (do not compare across metric switches).

Naming-history note
-------------------
The legacy stage-1 monolith called this "Phase E" (the CKA-distance
slot in the A → B → C → D → E → F phase chain). The current plugin
architecture has no phase taxonomy. Log strings retain
``"Stage 1 Phase E"`` for dashboard back-compat; new prose drops the
labels.
"""

from __future__ import annotations

import logging
import math

import torch

from ...utils.activation_hooks import ExpertOutputAccumulator
from ...utils.model_io import MATRIX_NAMES, build_banks
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)

_CKA_EPSILON = 1e-12                  # numerical floor for HSIC denominators
_CKA_M_MIN_VECTORIZED_FLOOR = 32      # below this, the GPU uniform-m path is unsafe
_SIMILARITY_METRIC_DEFAULT = "cka"


class CKADistancePlugin:
    """Per-layer CKA pairwise distance-matrix builder.

    Reads the calibration-pass-populated expert-output reservoir
    (``output_acc``) and the MoE layer list, computes the per-layer
    pairwise CKA distance matrix ``D = 1 − CKA``, and writes the
    per-layer dict back to the context for the downstream GRAPE merge
    plugin. Supports a weight-space ablation override
    (``stage1_grape.similarity_metric`` in {``"cka"``, ``"cosine"``, ``"mse"``})
    via :func:`_pairwise_distance_matrix`.

    See the module docstring for the dual paper citation (Kornblith CKA
    primitive + GRAPE consumer with the D-cka-distance sign-flip
    deviation), the official-code reference for the CKA primitive, and
    the two implementation paths (GPU vectorized / CPU per-pair).
    """

    name: str = "cka_distance"
    paper: str = (
        "CKA primitive: Kornblith et al. ICML 2019 — arXiv:1905.00414 "
        "(faithfully implemented; reference code "
        "google-research/google-research/representation_similarity @ "
        "89e3921863e276cdbe49bd25077905f75e981f4e). "
        "Consumer context: GRAPE (Zhang et al., arXiv:2604.06542) defines "
        "D^l as a similarity matrix; this plugin emits D^l in distance "
        "form (1 − CKA) with downstream argmin — deviation "
        "D-cka-distance (sign-flip; mathematically equivalent layer / "
        "pair ranking). No official code for GRAPE published. See module "
        "docstring for the full deviation derivation."
    )
    config_key: str = "stage1_grape"
    reads: tuple[str, ...] = (
        "output_acc",
        "moe_layers",
        "config",
    )
    writes: tuple[str, ...] = (
        "D_matrices",
    )
    # The per-(layer, expert) output reservoir (Phase B's ExpertOutputAccumulator).
    # Declared here so sub-task 10's orchestrator can wire the accumulator from the
    # CalibrationEngine. In this sub-task the legacy Phase B still populates the
    # accumulator inline; the plugin only consumes it via ``ctx.get("output_acc")``.
    provides: tuple[str, ...] = (
        "output_reservoir",
    )

    def is_enabled(self, config: dict) -> bool:
        """CKA distance computation is mandatory — every Stage 1 run executes Phase E.

        The ``similarity_metric`` config knob selects between CKA and the
        weight-space fallbacks (cosine/mse) per-layer; it does **not** disable
        the plugin.
        """
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Execute Phase E end-to-end.

        Reads slots ``output_acc``, ``moe_layers``, ``config`` from ``ctx``;
        writes ``D_matrices`` (dict[int, torch.Tensor]) back.

        Dispatches per-layer between :func:`_cka_distance_matrix` (default,
        consumes the Phase-B output reservoir) and
        :func:`_pairwise_distance_matrix` (weight-space ablation override
        when ``config["stage1_grape"]["similarity_metric"] != "cka"``).
        """
        output_acc: ExpertOutputAccumulator = ctx.get("output_acc")
        moe_layers = ctx.get("moe_layers")
        config: dict = ctx.get("config")
        s1 = config["stage1_grape"]
        metric = s1.get("similarity_metric", _SIMILARITY_METRIC_DEFAULT)

        log.info("Stage 1 Phase E: computing CKA pairwise distance matrices (D = 1 - CKA)")

        D_matrices: dict[int, torch.Tensor] = {}
        for k, ref in enumerate(moe_layers):
            if metric == _SIMILARITY_METRIC_DEFAULT:
                D = _cka_distance_matrix(output_acc, ref)
            else:
                D = _pairwise_distance_matrix(ref, metric=metric)
            D_matrices[ref.layer_idx] = D
            log.info(
                "  %s matrix: layer %d/%d (idx=%d)",
                "CKA" if metric == _SIMILARITY_METRIC_DEFAULT else metric,
                k + 1, len(moe_layers), ref.layer_idx,
            )

        if metric != _SIMILARITY_METRIC_DEFAULT:
            log.info(
                "Stage 1: overriding %s with weight-space metric '%s' (ablation mode)",
                _SIMILARITY_METRIC_DEFAULT, metric,
            )

        ctx.set("D_matrices", D_matrices)

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Phase E does not contribute to any JSON artifact.

        ``D_matrices`` is consumed in-memory by Phase F (GRAPE merge); it is
        never written to disk. Returning ``{}`` signals "no contribution" to
        the sub-task 10 orchestrator's :class:`ArtifactBuilder`.

        Returns
        -------
        dict
            Empty dict.
        """
        return {}


# ---------------------------------------------------------------------------
# CKA distance matrix from collected expert output representations
# ---------------------------------------------------------------------------


def _cka_distance_matrix(
    output_acc: ExpertOutputAccumulator,
    layer_ref,
) -> torch.Tensor:
    """Compute pairwise CKA distance matrix for all experts in a layer.

    Uses expert output representations collected during the calibration
    forward pass. CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    where HSIC uses linear kernels and the biased centering of Gretton (2005):
    K_c = K - row_mean - col_mean + grand_mean. Distance = (1 − CKA), clamped
    to [0, 1].

    Dispatches between two implementations based on the reservoir fill across
    the active expert set:

    - **GPU vectorized** (default for prod): subsamples every active expert
      to a single m_min over the active set, batches the Gram matrices, and
      computes the full N×N HSIC table in O(N · m² · d) GPU work. ~1 sec/layer
      on H200 vs ~10 min/layer for the CPU per-pair path.
    - **CPU per-pair fallback**: original implementation. Activated when the
      vectorized path is unsafe — m_min < 32 OR m_min < m_max // 4. Used by
      tests with tiny calibration sets and as a safety net when reservoir
      under-fill would force every pair to use a low m.

    With the prod default of ``num_calibration_samples=1024`` (the Stage 1
    YAML sub-config) and the ExpertOutputAccumulator reservoir cap of 256
    tokens/expert, all active
    experts saturate at m=256 and the GPU path is bit-equivalent (within fp32
    tolerance) to the original.

    Unactivated experts (m_e ≤ 1) get distance 1.0 in their full row and
    column, preserving the original placeholder semantics in both paths.
    """
    n_experts = layer_ref.num_routed_experts
    li = layer_ref.layer_idx

    # Pre-pass: gather active reservoirs and decide which path to take.
    active_indices: list[int] = []
    active_reprs: list[torch.Tensor] = []
    active_lengths: list[int] = []
    for e in range(n_experts):
        R = output_acc.get_representations(li, e)  # [m_e, d_out] CPU fp32 or None
        if R is None or R.shape[0] < 2:
            continue
        active_indices.append(e)
        active_reprs.append(R.detach().to(torch.float32))
        active_lengths.append(R.shape[0])

    # Initialize result: max dissimilarity 1.0, self-distance 0.0. Inactive
    # experts retain their full row/col at 1.0 — bit-identical to the original
    # zero-placeholder behavior.
    dist = torch.ones(n_experts, n_experts, dtype=torch.float32)
    dist.fill_diagonal_(0.0)

    if len(active_indices) < 2:
        return dist

    m_min = min(active_lengths)
    m_max = max(active_lengths)
    if m_min < _CKA_M_MIN_VECTORIZED_FLOOR or m_min < m_max // 4:
        # Reservoir is under-filled or skewed enough that uniform m_min would
        # silently degrade every pair's CKA precision. Fall back to the
        # per-pair m_common path so each pair retains its full intersection.
        log.info(
            "_cka_distance_matrix: layer %d active reservoir lengths span [%d..%d]; "
            "below vectorized floor (m_min ≥ %d and ≥ m_max//4) — falling back to "
            "CPU per-pair m_common path. Cause: small calibration / routing imbalance.",
            li, m_min, m_max, _CKA_M_MIN_VECTORIZED_FLOOR,
        )
        return _cka_distance_matrix_cpu_per_pair(
            active_indices, active_reprs, active_lengths, n_experts, dist
        )

    # ----- GPU vectorized path (uniform m_min over the active set) -----
    # Uniform-stride subsample to m_min — spreads token coverage across the
    # reservoir rather than front-slicing, identical to the original.
    X_list: list[torch.Tensor] = []
    for R in active_reprs:
        m = R.shape[0]
        if m == m_min:
            X_list.append(R)
        else:
            step = m / m_min
            idx = [int(k * step) for k in range(m_min)]
            X_list.append(R[idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.stack(X_list, dim=0).to(device)  # [N_active, m_min, d_out]

    # Batched linear-kernel Gram matrices: K_e = X_e @ X_e.T.
    K = torch.bmm(X, X.transpose(-2, -1))  # [N_active, m_min, m_min]

    # Biased HSIC centering: K_c = K - row_mean - col_mean + grand_mean.
    row_mean = K.mean(dim=2, keepdim=True)            # [N, m, 1]
    col_mean = K.mean(dim=1, keepdim=True)            # [N, 1, m]
    grand_mean = K.mean(dim=(1, 2), keepdim=True)     # [N, 1, 1]
    Kc = K - row_mean - col_mean + grand_mean

    # HSIC matrix H[i,j] = ⟨Kc_i, Kc_j⟩_F = (Kc_flat @ Kc_flat^T)[i,j].
    Kc_flat = Kc.reshape(Kc.shape[0], -1)             # [N, m*m]
    H = Kc_flat @ Kc_flat.t()                         # [N, N]

    # CKA = H[i,j] / sqrt(max(H[i,i], ε) · max(H[j,j], ε)).
    diag = H.diagonal().clamp(min=_CKA_EPSILON)
    norm = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    CKA = H / norm
    D_active = (1.0 - CKA).clamp(0.0, 1.0)
    D_active.fill_diagonal_(0.0)

    # Scatter the active-active sub-block back into the full distance matrix.
    idx_t = torch.tensor(active_indices, dtype=torch.long)
    dist[idx_t.unsqueeze(1), idx_t.unsqueeze(0)] = D_active.detach().cpu()

    return dist


def _cka_distance_matrix_cpu_per_pair(
    active_indices: list[int],
    active_reprs: list[torch.Tensor],
    active_lengths: list[int],
    n_experts: int,
    dist: torch.Tensor,
) -> torch.Tensor:
    """CPU per-pair m_common fallback. Preserves the original O(N²) Python loop
    used when the vectorized GPU path is unsafe (very small reservoirs / skewed
    fill). Fills the active-active sub-block of ``dist``; inactive rows/cols
    retain their pre-initialized 1.0 (with diagonal 0.0)."""
    n_active = len(active_indices)
    for ii in range(n_active):
        ei = active_indices[ii]
        Xi = active_reprs[ii]
        mi = active_lengths[ii]
        for jj in range(ii + 1, n_active):
            ej = active_indices[jj]
            Xj = active_reprs[jj]
            mj = active_lengths[jj]
            m_common = min(mi, mj)
            if m_common <= 1:
                # H=0 → CKA undefined → maximum distance.
                dist[ei, ej] = dist[ej, ei] = 1.0
                continue
            if mi > m_common:
                step = mi / m_common
                Xi_c = Xi[[int(k * step) for k in range(m_common)]]
            else:
                Xi_c = Xi
            if mj > m_common:
                step = mj / m_common
                Xj_c = Xj[[int(k * step) for k in range(m_common)]]
            else:
                Xj_c = Xj
            # Biased HSIC centering (Gretton 2005), identical to the GPU path.
            Ki_raw = Xi_c @ Xi_c.T
            Ki = Ki_raw - Ki_raw.mean(dim=1, keepdim=True) - Ki_raw.mean(dim=0, keepdim=True) + Ki_raw.mean()
            Kj_raw = Xj_c @ Xj_c.T
            Kj = Kj_raw - Kj_raw.mean(dim=1, keepdim=True) - Kj_raw.mean(dim=0, keepdim=True) + Kj_raw.mean()
            hsic_ij = float((Ki * Kj).sum().item())
            hsic_ii = float((Ki * Ki).sum().item())
            hsic_jj = float((Kj * Kj).sum().item())
            denom = math.sqrt(max(hsic_ii, _CKA_EPSILON) * max(hsic_jj, _CKA_EPSILON))
            cka = hsic_ij / denom
            d = max(0.0, min(1.0, 1.0 - cka))
            dist[ei, ej] = d
            dist[ej, ei] = d
    # Diagonal is already 0.0 from the caller.
    return dist


# ---------------------------------------------------------------------------
# Weight-space distance matrix fallback (for ablation / testing)
# ---------------------------------------------------------------------------


# Scale note (A-C-N-1): cosine is scaled to [0, 1] by (1 - sim) / 2; MSE is normalized by
# its max. The two scales are NOT directly comparable across runs that switch metrics.
def _pairwise_distance_matrix(layer_ref, *, metric: str) -> torch.Tensor:
    """Weight-space pairwise distance matrix (fallback for ablation)."""
    banks = build_banks(layer_ref)
    vecs: list[torch.Tensor] = []
    for e in range(layer_ref.num_routed_experts):
        parts = [banks[name].get(e).detach().to(torch.float32).flatten()
                 for name in MATRIX_NAMES]
        vecs.append(torch.cat(parts))
    if not vecs:
        return torch.zeros(0, 0)
    W = torch.stack(vecs)
    # Cosine distance is in [0, 1]; MSE distance is normalised to [0, 1] by dividing by
    # its max; if all experts are identical (max=0), clamp keeps denominator at 1e-8 and
    # all distances stay near zero. These two modes produce incommensurable R values;
    # do not compare across metric runs.
    if metric == "cosine":
        W = torch.nn.functional.normalize(W, dim=1)
        sim = W @ W.transpose(0, 1)
        dist = (1.0 - sim).clamp(min=0.0, max=2.0) / 2.0
    elif metric == "mse":
        sq = (W * W).sum(dim=1)
        dot = W @ W.transpose(0, 1)
        dist = (sq[:, None] + sq[None, :] - 2 * dot).clamp(min=0.0)
        dist = dist / (dist.max().clamp(min=1e-8))
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    return dist
