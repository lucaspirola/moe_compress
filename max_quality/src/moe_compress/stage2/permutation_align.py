"""Stage 2 permutation-alignment primitives.

Extracted from ``stage2_reap_ream.py`` in Task 4 of the plugin-architecture
refactor. Houses the three symbols cost-matrix plugins need without dragging
in the heavyweight merge engine:

  * ``_PermAlignCache`` -- per-layer (perm, residual) cache keyed by
    ``(layer_idx, centroid_id, noncentroid_id)``. Spec § 5 step 4T(c)(i)-(ii).
  * ``_aligned_whitened_residual`` -- three-term whitened Frobenius residual
    under a fixed Hungarian permutation. Spec § 5 step 4T(c)(ii).
  * ``_permutation_align_to_centroid`` -- Hungarian alignment of a child
    expert's neuron axis to a centroid (optionally weighted by activation
    means; F2-PERM-ALIGN-NORM, PERM-ACT-SCALE, B-C-M-1).

``stage2_reap_ream`` re-imports all three at module scope so existing call
sites and tests keep working unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class _PermAlignCache:
    """Per-layer cache of Hungarian permutations and whitened residuals.

    Stage 2 v2 spec § 5 step 4T(c)(i)–(ii) (M1, "reuse merge-time Hungarian
    for the assignment cost"): the cost matrix and the merge step share the
    same per-pair Hungarian alignment. This cache lets both consumers see
    the result of one computation.

    Keys: ``(layer_idx, centroid_id, noncentroid_id)``.
    Values: ``(perm: np.ndarray, residual: float | None)``. ``residual`` is
    ``None`` when the cache entry came from the legacy v1 merge path (which
    only knows the permutation, not the whitened residual).

    Cleared at the start of every layer; bounded by ``N × K`` per layer
    (default 256 × 48 = 12,288 entries × ~512 bytes/perm ≈ 6 MB).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[int, int, int], tuple[np.ndarray, float | None]] = {}

    def get(self, key: tuple[int, int, int]) -> tuple[np.ndarray, float | None] | None:
        return self._store.get(key)

    def put(self, key: tuple[int, int, int], perm: np.ndarray, residual: float | None) -> None:
        self._store[key] = (perm, residual)

    def has(self, key: tuple[int, int, int]) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _aligned_whitened_residual(
    *,
    ref_gate: torch.Tensor,
    ref_up: torch.Tensor,
    ref_down: torch.Tensor,
    child_gate: torch.Tensor,
    child_up: torch.Tensor,
    child_down: torch.Tensor,
    perm: np.ndarray,
    a_sqrt_gate_up: torch.Tensor,
    a_sqrt_down: torch.Tensor,
    whitening_mode: str,
) -> float:
    """Three-term whitened Frobenius residual under a fixed permutation.

    Per spec § 5 step 4T(c)(ii):
        R_cm = ‖(W_c_gate − W_m_gate[perm, :]) · A_gate_up^{1/2}‖_F
             + ‖(W_c_up   − W_m_up[perm, :])   · A_gate_up^{1/2}‖_F
             + ‖(W_c_down − W_m_down[:, perm]) · A_down^{1/2}    ‖_F

    The whitening factor multiplies ΔW on the **right** (input axis), per the
    AA-SVD lineage and the Round-1 spec-review dimensional fix.

    Convention (PyTorch nn.Linear weight shapes):
        W_gate, W_up : (d_int, hidden)
        W_down       : (hidden, d_int)
        A_gate_up    : (hidden, hidden)
        A_down       : (d_int, d_int)
        perm         : length d_int — child neurons reordered to align with the centroid.
    """
    # Import here so the module load order doesn't depend on cov_sqrt being
    # available (cov_sqrt itself depends only on torch, no circular risk).
    from ..utils.cov_sqrt import whitened_residual

    perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)

    # Aligned child weights (gate / up / down). All three projections need
    # the same per-pair permutation applied on the d_int axis.
    aligned_gate = child_gate[perm_t, :]      # (d_int, hidden)
    aligned_up   = child_up[perm_t, :]        # (d_int, hidden)
    aligned_down = child_down[:, perm_t]      # (hidden, d_int)

    delta_gate = ref_gate - aligned_gate      # (d_int, hidden)
    delta_up   = ref_up   - aligned_up        # (d_int, hidden)
    delta_down = ref_down - aligned_down      # (hidden, d_int)

    r_gate = whitened_residual(delta_gate, a_sqrt_gate_up, mode=whitening_mode)
    r_up   = whitened_residual(delta_up,   a_sqrt_gate_up, mode=whitening_mode)
    r_down = whitened_residual(delta_down, a_sqrt_down,    mode=whitening_mode)

    return float(r_gate + r_up + r_down)


def _permutation_align_to_centroid(
    ref_gate: torch.Tensor,
    ref_up: torch.Tensor,
    child_gate: torch.Tensor,
    child_up: torch.Tensor,
    ref_act_mean: torch.Tensor | None = None,
    child_act_mean: torch.Tensor | None = None,
) -> np.ndarray:
    def _safe_norm(M):
        # B-C-L-2: when M is all-zero (or constant), m_max == m_min and we fall
        # through to torch.zeros_like(M). This means a zero-distance pair stays
        # zero (no cost contribution from that component) — the desired behavior
        # for Hungarian assignment where ties resolve arbitrarily.
        m_min = float(M.min())
        m_max = float(M.max())
        if m_max > m_min:
            return (M - m_min) / (m_max - m_min)
        return torch.zeros_like(M)

    # Keep cost-matrix construction on the device of the input weights — the
    # explicit .cpu() calls present here previously forced ~50-100 ms of CPU
    # cdist per pair-alignment, vs ~1 ms on GPU; with up to ~5K calls/layer ×
    # 40 layers the regression compounded to >10 min/run. The single CPU sync
    # is deferred to the Hungarian step below, which is unavoidably CPU
    # (scipy.optimize.linear_sum_assignment).
    # All inputs must share the same device (callers stage tensors via
    # build_banks(layer_ref), which keys off the live model device); cdist
    # would error on mixed-device inputs.
    # torch.cdist does not support bfloat16 on CPU (CUDA-only). Upcast here
    # to fp32 for portability — cdist is on small (d_int × d_int) matrices
    # so the upcast cost is negligible (cdist is not the B2-targeted hot
    # path; the per-pair gain comes from the merge math and the perm-apply
    # on the larger (hidden × intermediate) matrices, both of which remain
    # in native dtype).
    C_gate = torch.cdist(ref_gate.float(), child_gate.float())
    C_up   = torch.cdist(ref_up.float(),   child_up.float())
    if ref_act_mean is not None and child_act_mean is not None:
        # L2-normalize both activation-mean vectors along the neuron dimension
        # before computing L2 distance (spec §5, F2-PERM-ALIGN-NORM).
        # eps=1e-8 guards against zero-norm vectors (all-zero activations);
        # F.normalize returns a zero vector for those, which is the safest
        # fallback (zero-norm input → zero output, no NaN).
        # Move act_mean tensors to the same device as the weight tensors;
        # they originate from CPU storage in ReamCostAccumulator._neuron_act_sum
        # but ref_gate/child_gate live on the model's device. Without this
        # explicit move, C_act lands on CPU while C_wt is on GPU, and the
        # subsequent `C = C_act + C_wt` raises a device-mismatch RuntimeError.
        _act_device = ref_gate.device
        ref_act_n   = torch.nn.functional.normalize(ref_act_mean.float().to(_act_device),   p=2, dim=0, eps=1e-8)
        child_act_n = torch.nn.functional.normalize(child_act_mean.float().to(_act_device), p=2, dim=0, eps=1e-8)
        C_act = torch.cdist(
            ref_act_n.unsqueeze(-1),
            child_act_n.unsqueeze(-1),
        )
        # Scale each cost component to [0, 1] before summing so that
        # L2-normalized activation distances (O(1/√d_ffn)) are not
        # negligible relative to gate/up weight distances (O(√d_hidden))
        # — spec §5, PERM-ACT-SCALE.
        # B-C-M-1: spec §5 / D5b defines C = C_act + C_wt where C_wt is the
        # gate+up Frobenius distance treated as a SINGLE component (sum first,
        # then normalize once), not two separately-normalized components.
        C_act = _safe_norm(C_act)
        C_wt = _safe_norm(C_gate + C_up)
        C = C_act + C_wt
    else:
        # B-C-M-1: same single-component treatment for the no-activation path.
        C = _safe_norm(C_gate + C_up)
    # Hungarian solver requires CPU numpy — single sync at the end.
    # C.float(): scipy.optimize.linear_sum_assignment does not accept
    # bfloat16, AND torch.cdist on CPU does not implement bfloat16
    # (CUDA-only). Both .float() casts (here and at the cdist inputs
    # above) preserve portability for CPU bf16 inputs while letting
    # the merge math in _tentative_merged_weights stay in native dtype.
    # No-op for existing fp32 callers. Added for SC_FAST_PLAN_V3 §4-B2.
    _, col_ind = linear_sum_assignment(C.float().detach().cpu().numpy())
    return col_ind
