"""Muon optimizer — momentum + Newton-Schulz orthogonalization for 2D weights.

A small, single-device implementation copied into the project (moe_compress is
standalone, so there is no external Muon dependency). Used by the Stage-2
per-layer merge-heal step, which fine-tunes 2D matrices only — per-centroid
expert projections ``(d_int, hidden)`` / ``(hidden, d_int)`` and the routed
``router.weight`` ``(n_kept, hidden)``.

Muon (Keller Jordan, 2024) takes the SGD-momentum update for a weight *matrix*
and replaces it with its nearest semi-orthogonal matrix, computed by a
quintic Newton-Schulz iteration. Versus AdamW it keeps **one** momentum buffer
per parameter (not two), and empirically converges faster on matrix weights.

Scope / limitations:
* 2D parameters only. The Newton-Schulz iteration is defined on matrices; 1D
  parameters (biases, norm scales) must be optimized by a separate AdamW group.
  ``Muon.step`` raises if it is handed a non-2D parameter with a gradient.
* Single device. No distributed / sharded update path.
"""
from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Return an approximately semi-orthogonal matrix sharing G's row/col space.

    Runs the quintic Newton-Schulz iteration of Keller Jordan's Muon. The
    iteration is performed in bfloat16 (the quintic is numerically stable there
    and it halves the matmul cost); the caller converts back to the parameter
    dtype. The quintic coefficients ``(3.4445, -4.7750, 2.0315)`` are tuned so
    the iteration's fixed point has singular values in ``[~0.7, ~1.3]`` — the
    update is *approximately* orthogonal, which is all Muon needs.

    Args:
        G: a 2D gradient/momentum matrix.
        steps: number of Newton-Schulz iterations (5 is the standard default).
        eps: guards the initial spectral-norm normalization against G == 0.

    Returns:
        A bfloat16 matrix of the same shape as G.
    """
    if G.ndim != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects a 2D matrix, got ndim={G.ndim}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # The iteration below assumes #rows <= #cols (it forms X @ X.T). Transpose
    # tall matrices, then transpose the result back.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    # Normalize so the spectral norm is <= 1, the convergence basin of the
    # iteration. Frobenius norm >= spectral norm, so dividing by it is safe.
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: orthogonalized-momentum optimizer for 2D weight matrices.

    Per step, per parameter: update the momentum buffer, optionally apply a
    Nesterov look-ahead, orthogonalize the resulting matrix via
    :func:`zeropower_via_newtonschulz5`, then take a step scaled by
    ``sqrt(max(1, rows / cols))`` — the standard Muon adjustment that keeps the
    update RMS consistent across non-square matrices.

    Args:
        params: iterable of 2D parameters (or param groups).
        lr: learning rate.
        momentum: momentum coefficient for the gradient buffer.
        nesterov: if True, use the Nesterov look-ahead update.
        ns_steps: Newton-Schulz iteration count.
        weight_decay: decoupled (AdamW-style) weight decay; 0.0 disables it.
    """

    def __init__(
        self,
        params,
        lr: float = 2.0e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 — see class docstring
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.ndim != 2:
                    raise ValueError(
                        "Muon only optimizes 2D parameters; got a parameter with "
                        f"ndim={grad.ndim}. Route 1D params (biases, norm scales) "
                        "to a separate AdamW group."
                    )

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Decoupled weight decay (applied to the parameter, not via grad).
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # sqrt(max(1, rows/cols)) keeps the per-element update RMS
                # consistent for non-square matrices (standard Muon scaling).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)

        return loss
