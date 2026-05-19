"""Unit tests for the Muon optimizer (``moe_compress.utils.muon``).

Covers:
- ``zeropower_via_newtonschulz5`` produces a well-conditioned (approximately
  semi-orthogonal) matrix — singular values collapse toward 1 — and preserves
  shape for both wide and tall inputs.
- ``Muon`` reduces a toy least-squares objective.
- ``Muon`` rejects non-2D parameters (they belong in an AdamW group).
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.muon import Muon, zeropower_via_newtonschulz5


# ---------------------------------------------------------------------------
# zeropower_via_newtonschulz5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(16, 16), (12, 48), (48, 12)])
def test_newtonschulz_orthogonalizes(shape):
    """A random matrix should come out well-conditioned: 5 quintic steps pull
    every singular value into a tight band around 1 and collapse the spread."""
    torch.manual_seed(0)
    G = torch.randn(shape)
    sv_in = torch.linalg.svdvals(G.float())

    X = zeropower_via_newtonschulz5(G, steps=5)

    assert X.shape == G.shape
    sv = torch.linalg.svdvals(X.float())
    # The quintic's fixed point lands every singular value near 1.
    assert sv.max().item() < 1.6
    assert sv.min().item() > 0.5
    # The spectrum is much tighter than the input's.
    assert (sv.max() / sv.min()).item() < (sv_in.max() / sv_in.min()).item()


def test_newtonschulz_collapses_ill_conditioned_spectrum():
    """Even a condition-number-~1000 matrix has its spectrum collapsed by an
    order of magnitude (5 steps won't fully orthogonalize it, but must tighten
    it dramatically)."""
    torch.manual_seed(1)
    r = 16
    u, _ = torch.linalg.qr(torch.randn(r, r))
    v, _ = torch.linalg.qr(torch.randn(r, r))
    G = u @ torch.diag(torch.logspace(-3, 0, r)) @ v.T  # condition number ~1000

    sv = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=5).float())
    assert (sv.max() / sv.min()).item() < 10.0  # ~1000 → < 10


def test_newtonschulz_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        zeropower_via_newtonschulz5(torch.randn(8))


def test_newtonschulz_handles_zero_matrix():
    """The eps guard must keep a zero input finite (no division by zero)."""
    X = zeropower_via_newtonschulz5(torch.zeros(8, 8), steps=5)
    assert torch.isfinite(X.float()).all()


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------


def test_muon_minimizes_least_squares():
    """Muon should drive W toward a target on a convex matrix objective."""
    torch.manual_seed(0)
    target = torch.randn(32, 16)
    W = torch.nn.Parameter(torch.zeros(32, 16))
    opt = Muon([W], lr=0.05, momentum=0.95, nesterov=True)

    initial_loss = torch.mean((W - target) ** 2).item()
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((W - target) ** 2)
        loss.backward()
        opt.step()
    final_loss = torch.mean((W - target) ** 2).item()

    assert final_loss < 0.2 * initial_loss


def test_muon_rejects_1d_parameter():
    """A 1D parameter with a gradient must raise — biases go to AdamW."""
    p = torch.nn.Parameter(torch.zeros(8))
    opt = Muon([p], lr=0.01)
    p.grad = torch.randn(8)
    with pytest.raises(ValueError, match="2D"):
        opt.step()


def test_muon_skips_params_without_grad():
    """A parameter with grad=None must be skipped, not raise."""
    W = torch.nn.Parameter(torch.zeros(4, 4))
    opt = Muon([W], lr=0.01)
    before = W.detach().clone()
    opt.step()  # no backward called → grad is None
    assert torch.equal(W.detach(), before)


def test_muon_weight_decay_shrinks_idle_param():
    """Decoupled weight decay must shrink a parameter even on a zero gradient."""
    W = torch.nn.Parameter(torch.ones(4, 4))
    opt = Muon([W], lr=0.1, weight_decay=0.5)
    W.grad = torch.zeros(4, 4)
    opt.step()
    # zero grad → orthogonalized update is ~0; only decay (1 - lr*wd) applies.
    assert W.detach().abs().max().item() < 1.0
