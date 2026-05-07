"""Activation-aware covariance whitening helpers for Stage 2 v2.

This module provides the eigen-square-root of an input-covariance matrix used
to weight the Stage 2 v2 post-alignment cost matrix per the AA-SVD lineage
(arXiv 2604.02119; cited by upstream ALGORITHM_REFERENCE.md § 6).

The cost being minimized at the merge step is::

    E_x ‖ΔW · x‖² = tr(ΔW · A · ΔW^T) = ‖ΔW · A^{1/2}‖_F²

where ``A = E_x[x x^T]`` is the input covariance over calibration tokens. The
whitening factor ``A^{1/2}`` therefore multiplies ``ΔW`` on the **right**
(input axis), never the left — this is the dimensional fix called out in the
spec-vs-papers review (Round 1, F2/F3).

For Qwen3.6-35B-A3B::

    A_gate_up : (hidden, hidden)        = (2048, 2048)
    A_down    : (d_int,  d_int)         = ( 512,  512)

Two whitening modes are supported:

* ``"diag"`` — return the per-channel sqrt vector ``sqrt(diag(A))``. Cheap
  fallback for memory-pressured runs. Right-multiplying ``ΔW`` by the diag
  vector is element-wise column scaling.
* ``"full"`` — return ``V · diag(sqrt(λ_clamped)) · V^T`` via
  ``torch.linalg.eigh``. The eigenvalue clamp mirrors Stage 3's
  ``_precompute_eigh`` noise-floor convention (``stage3_svd.py:1485``).

A small in-process cache keyed on ``(layer_idx, expert_idx, matrix_name,
mode)`` avoids recomputing the eigen-decomposition when both Stage 2 v2 (cost
matrix) and the merge step would otherwise duplicate it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch

# Mirrors stage3_svd._NOISE_FLOOR_BY_DTYPE intent: clamp eigenvalues that fall
# below σ_max * rel_floor to suppress numerical noise that would otherwise
# blow up sqrt(λ) for nearly-singular A.
_REL_NOISE_FLOOR: dict[torch.dtype, float] = {
    torch.float32: 1e-6,
    torch.float64: 1e-12,
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
}

WhiteningMode = Literal["none", "diag", "full"]


class CovSqrtCache:
    """Bounded in-process cache of computed A^{1/2} tensors.

    The cache is keyed on ``(layer_idx, expert_idx, matrix_name, mode)`` so
    layer/expert pairs are isolated and the same matrix can coexist as both
    the diag and full forms for ablation A/B comparison.

    The cache is intentionally small (default 256 entries) because each
    full eigen-sqrt at hidden=2048 is ~16 MB on CPU; 256 entries ≈ 4 GB is a
    soft ceiling that fits Stage 2's host-RAM headroom comfortably.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._store: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: tuple) -> torch.Tensor | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: tuple, value: torch.Tensor) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def compute_a_sqrt(
    A: torch.Tensor,
    *,
    mode: WhiteningMode,
    eigvalue_floor: float | None = None,
) -> torch.Tensor:
    """Return the activation-whitening factor for input covariance ``A``.

    Args:
        A: Square symmetric input-covariance matrix, shape ``(d, d)``.
        mode: ``"none"`` (returns identity-equivalent), ``"diag"`` (returns
            ``sqrt(diag(A))`` as a 1-D vector of length ``d`` — apply by
            right-multiplying ``ΔW * vec`` element-wise on the column axis),
            or ``"full"`` (returns the symmetric matrix
            ``V · diag(sqrt(λ_clamped)) · V^T`` of shape ``(d, d)``).
        eigvalue_floor: optional absolute clamp for eigenvalues. If ``None``,
            uses the relative noise floor from ``_REL_NOISE_FLOOR`` keyed on
            ``A.dtype`` (matching ``stage3_svd._precompute_eigh``).

    Returns:
        Tensor on the same device/dtype as ``A``. For ``mode == "none"``,
        returns a scalar 1.0 sentinel (callers should treat as a no-op).
    """
    if mode == "none":
        return torch.tensor(1.0, dtype=A.dtype, device=A.device)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"compute_a_sqrt: expected square 2-D tensor, got shape {tuple(A.shape)}"
        )

    if mode == "diag":
        diag = torch.diagonal(A).clamp_min(0.0)
        return torch.sqrt(diag)

    if mode == "full":
        # eigh expects symmetric input; cast to fp32 for numerical stability
        # of the eigen-decomposition then cast back to the original dtype on
        # the way out. Mirrors stage3_svd._precompute_eigh.
        A_work = A.to(torch.float32)
        eigvals, eigvecs = torch.linalg.eigh(A_work)
        sigma_max = float(eigvals[-1].clamp_min(0).item())
        rel_floor = (
            eigvalue_floor
            if eigvalue_floor is not None
            else _REL_NOISE_FLOOR.get(A.dtype, 1e-6)
        )
        thresh = max(sigma_max * rel_floor, 1e-12)
        eigvals_clamped = eigvals.clamp_min(thresh)
        sqrt_lam = torch.sqrt(eigvals_clamped)
        a_sqrt = (eigvecs * sqrt_lam.unsqueeze(0)) @ eigvecs.transpose(-1, -2)
        return a_sqrt.to(A.dtype)

    raise ValueError(f"compute_a_sqrt: unknown mode {mode!r}")


def whitened_residual(
    delta_w: torch.Tensor,
    a_sqrt: torch.Tensor,
    *,
    mode: WhiteningMode,
) -> torch.Tensor:
    """Compute ``‖ΔW · A^{1/2}‖_F`` with the whitening on the **right** of ΔW.

    Per the AA-SVD lineage and the Round-1 spec-review fix, the whitening
    factor multiplies ``ΔW`` on the input (column) axis, never on the left.

    Args:
        delta_w: ``(out, in)`` weight delta tensor.
        a_sqrt: output of :func:`compute_a_sqrt`. Either a 1-D vector (diag
            mode) of length ``in``, a square ``(in, in)`` matrix (full mode),
            or a scalar 1.0 (none mode → returns ``‖ΔW‖_F`` as-is).
        mode: must match the mode used to build ``a_sqrt``.

    Returns:
        Scalar tensor — the Frobenius norm of the whitened residual.
    """
    if mode == "none":
        return torch.linalg.matrix_norm(delta_w, ord="fro")

    if mode == "diag":
        if a_sqrt.ndim != 1:
            raise ValueError(
                f"whitened_residual(diag): expected 1-D a_sqrt, got shape "
                f"{tuple(a_sqrt.shape)}"
            )
        if a_sqrt.shape[0] != delta_w.shape[-1]:
            raise ValueError(
                f"whitened_residual(diag): a_sqrt dim {a_sqrt.shape[0]} does "
                f"not match ΔW input axis {delta_w.shape[-1]}"
            )
        return torch.linalg.matrix_norm(delta_w * a_sqrt.to(delta_w.dtype), ord="fro")

    if mode == "full":
        if a_sqrt.ndim != 2 or a_sqrt.shape[0] != a_sqrt.shape[1]:
            raise ValueError(
                f"whitened_residual(full): expected square 2-D a_sqrt, got "
                f"shape {tuple(a_sqrt.shape)}"
            )
        if a_sqrt.shape[0] != delta_w.shape[-1]:
            raise ValueError(
                f"whitened_residual(full): a_sqrt dim {a_sqrt.shape[0]} does "
                f"not match ΔW input axis {delta_w.shape[-1]}"
            )
        return torch.linalg.matrix_norm(delta_w @ a_sqrt.to(delta_w.dtype), ord="fro")

    raise ValueError(f"whitened_residual: unknown mode {mode!r}")
