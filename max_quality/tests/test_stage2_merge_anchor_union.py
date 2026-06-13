"""Feature B — REAM merge-anchor union fix.

The post-merge covariance remap (`_remap_covariance_for_layer`) must copy the
**group UNION** ``Σ_j G_j`` of every merged member's input Gram into the
survivor slot, NOT the centroid's own Gram. For ``down_proj`` the union summand
must be permuted on BOTH axes by the EXACT merge permutation captured at
``merging.py:280/291/298`` (the same object that permuted ``Wm``), mirroring the
RegMean down-Gram permutation (``merging.py:352-360``).

Non-merged (singleton / protected) survivors stay byte-identical: with no
``grouped`` argument the function takes today's verbatim singleton path.
"""
import torch

from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.stage2.shared_io import _remap_covariance_for_layer


def _put(cov, li, e, name, G, ntok):
    cov.covariance[(li, e, name)] = G.clone()
    cov.token_count[(li, e, name)] = ntok


def test_survivor_anchor_is_group_union_gate():
    cov = InputCovarianceAccumulator()
    G = {e: torch.full((4, 4), float(e + 1)) for e in (0, 1, 2)}
    for e in (0, 1, 2):
        _put(cov, 7, e, "gate_proj", G[e], ntok=10 * (e + 1))
    _remap_covariance_for_layer(cov, 7, kept_ids=[0], grouped={0: [0, 1, 2]})
    A = cov.covariance[(7, 0, "gate_proj")]
    assert torch.equal(A, G[0] + G[1] + G[2]), "anchor must be group UNION Σ_j G_j"
    assert cov.token_count[(7, 0, "gate_proj")] == 10 + 20 + 30


def test_non_merged_survivor_unchanged_byte_identical():
    cov = InputCovarianceAccumulator()
    G = torch.arange(9.0).reshape(3, 3)
    _put(cov, 3, 5, "gate_proj", G, ntok=42)
    _remap_covariance_for_layer(cov, 3, kept_ids=[5], grouped={})
    assert torch.equal(cov.covariance[(3, 0, "gate_proj")], G)
    assert cov.token_count[(3, 0, "gate_proj")] == 42


def test_survivor_anchor_down_is_permuted_union():
    cov = InputCovarianceAccumulator()
    d = 3
    G = {
        e: torch.arange(d * d, dtype=torch.float32).reshape(d, d) + 100 * e
        for e in (0, 1, 2)
    }
    for e in (0, 1, 2):
        _put(cov, 2, e, "down_proj", G[e], ntok=5)
    perms = {0: None, 1: [2, 0, 1], 2: [1, 2, 0]}  # centroid perm = None (identity)

    def pb(t, p):
        idx = torch.as_tensor(p, dtype=torch.long)
        return t.index_select(0, idx).index_select(1, idx)

    expected = G[0] + pb(G[1], perms[1]) + pb(G[2], perms[2])
    _remap_covariance_for_layer(
        cov, 2, kept_ids=[0], grouped={0: [0, 1, 2]}, member_perms={0: perms},
    )
    assert torch.allclose(cov.covariance[(2, 0, "down_proj")], expected)
