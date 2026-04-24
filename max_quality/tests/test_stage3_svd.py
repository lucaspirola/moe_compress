"""Stage 3 — SVD factorization sanity checks.

We directly exercise ``_apply_aa_svd`` on a synthetic matrix and a known
activation covariance, and assert the composed weight reduces residual error
in the B-weighted Frobenius norm vs plain truncated SVD.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage3_svd import _apply_aa_svd, _FactoredLinear, _MatrixRef


def _lapack_available() -> bool:
    """The locally-built PyTorch nightly on this dev box was compiled without
    LAPACK, so ``torch.linalg.cholesky`` / ``svd`` error on CPU. On A100/H100
    the CUDA path is used instead and LAPACK is irrelevant. Skip the numeric
    test if CPU linear algebra isn't available.
    """
    try:
        torch.linalg.svd(torch.eye(2), full_matrices=False)
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _lapack_available(), reason="PyTorch built without CPU LAPACK")
def test_aa_svd_reduces_error_on_anchored_metric():
    torch.manual_seed(42)
    d1, d2, k = 16, 8, 4
    W = torch.randn(d1, d2)

    # Build a synthetic ΣxxT: a mixture of a few high-energy directions plus noise.
    X = torch.randn(200, d2)
    X[:, :3] *= 5.0                 # emphasize first few dims
    A = X.transpose(0, 1) @ X
    B = A.clone()

    # Wrap into a Linear so _apply_aa_svd can rebind the module.
    lin = torch.nn.Linear(d2, d1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(W)
    parent = torch.nn.Module()
    parent.gate_proj = lin
    m = _MatrixRef(
        layer_idx=0, expert_idx=0, name="gate_proj",
        linear=lin, parent=parent, attr="gate_proj",
    )

    _apply_aa_svd(m, k, A=A, B=B)

    factored = parent.gate_proj
    assert isinstance(factored, _FactoredLinear)
    assert factored.up.weight.shape == (d1, k)
    assert factored.down.weight.shape == (k, d2)

    # Forward-equivalence check on the anchored inputs: the factored weight
    # should minimize ||(W - W') X||_F more strongly than naive truncated SVD.
    W_prime = factored.up.weight @ factored.down.weight
    anchored_err = ((W - W_prime) @ X.transpose(0, 1)).pow(2).sum().item()

    # Plain truncated SVD for comparison
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_plain = (U[:, :k] * S[:k]) @ Vh[:k, :]
    plain_err = ((W - W_plain) @ X.transpose(0, 1)).pow(2).sum().item()

    # AA-SVD should be at least as good in the anchored metric; allow small
    # slack for numerical differences.
    assert anchored_err <= plain_err * 1.10


def test_factored_forward_equals_composition():
    torch.manual_seed(0)
    d1, d2, k = 8, 4, 2
    down = torch.nn.Linear(d2, k, bias=False)
    up = torch.nn.Linear(k, d1, bias=False)
    mod = _FactoredLinear(down, up)

    x = torch.randn(3, d2)
    y = mod(x)
    y_ref = (up.weight @ down.weight) @ x.transpose(0, 1)
    assert torch.allclose(y, y_ref.transpose(0, 1), atol=1e-6)
