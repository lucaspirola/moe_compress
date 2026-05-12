"""Tests for ``ReamCostAccumulator.finalize_batch`` — the vectorized
gated-output cosine-similarity accumulator that dominates Stage 2's per-batch
wall time on Qwen3.6-35B-A3B.

The vectorized rewrite landed to eliminate the per-batch Python-loop plumbing
(see plan ``the-oregon-b200-we-re-wondrous-scott.md``). These tests pin the
algebra against a slow, obviously-correct NumPy reference, covering the seven
fixtures called out in the plan.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch

from moe_compress.utils.activation_hooks import (
    _FINALIZE_BATCH_CHUNK,
    ReamCostAccumulator,
)


# ---------------------------------------------------------------------------
# Slow, obviously-correct NumPy reference for `gated_output_sim`
# ---------------------------------------------------------------------------
def _numpy_reference_gated_output_sim(
    per_expert_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[int, int], float]:
    """Reference numerator of REAM Eq. 8 — Σ_{t ∈ jointly-active(i,j)} cos(g_i[t], g_j[t]).

    Args:
        per_expert_data: ``{eid: (token_indices[T_e] int64, gated[T_e, d_hid] float)}``.

    Returns:
        ``{(e_i, e_j): sim_sum}`` for every ordered pair ``(e_i, e_j)`` with
        ``e_i != e_j`` and non-zero accumulated sum. Mirrors the symmetric
        write semantics of ``finalize_batch`` (both ``(i, j)`` and ``(j, i)``).
        Self-pairs ``(e, e)`` are excluded.
    """
    token_to_active: defaultdict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for eid, (indices, gated) in per_expert_data.items():
        for pos, t in enumerate(indices.tolist()):
            token_to_active[int(t)][eid] = gated[pos]

    sim_sum: defaultdict[tuple[int, int], float] = defaultdict(float)
    for active in token_to_active.values():
        eids = sorted(active.keys())
        if len(eids) < 2:
            continue
        for i_idx, e_i in enumerate(eids):
            v_i = active[e_i].astype(np.float64)
            n_i = float(np.linalg.norm(v_i))
            if n_i < 1e-12:
                continue
            for e_j in eids[i_idx + 1:]:
                v_j = active[e_j].astype(np.float64)
                n_j = float(np.linalg.norm(v_j))
                if n_j < 1e-12:
                    continue
                cos = float(np.dot(v_i, v_j) / (n_i * n_j))
                sim_sum[(e_i, e_j)] += cos
                sim_sum[(e_j, e_i)] += cos

    # Drop near-zero entries to match the `if v == 0.0: continue` filter in
    # finalize_batch's dict-merge.
    return {k: v for k, v in sim_sum.items() if v != 0.0}


def _build_acc(
    per_expert_data: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    num_experts: int,
    layer_idx: int = 0,
) -> ReamCostAccumulator:
    """Populate a fresh ReamCostAccumulator's ``_batch_gated_indexed`` directly."""
    acc = ReamCostAccumulator(num_experts=num_experts)
    for e, (indices, gated) in per_expert_data.items():
        acc._batch_gated_indexed[(layer_idx, e)] = (indices, gated)
    return acc


def _assert_matches_reference(
    acc: ReamCostAccumulator,
    expected: dict[tuple[int, int], float],
    *,
    layer_idx: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    actual = {
        (k[1], k[2]): v
        for k, v in acc.gated_output_sim.items()
        if k[0] == layer_idx
    }
    extra = set(actual.keys()) - set(expected.keys())
    missing = set(expected.keys()) - set(actual.keys())
    # Tolerate keys where the expected magnitude is below atol — they are
    # legitimately ambiguous (fp32 rounding can flip them across zero).
    extra = {k for k in extra if abs(actual[k]) > atol}
    missing = {k for k in missing if abs(expected[k]) > atol}
    assert not extra, f"unexpected keys with nontrivial values: {extra}"
    assert not missing, f"missing keys: {missing}"
    for k, exp_v in expected.items():
        if k not in actual:
            continue
        act_v = actual[k]
        tol = max(atol, rtol * max(1.0, abs(exp_v)))
        assert abs(act_v - exp_v) < tol, (
            f"pair {k}: expected {exp_v:.6e}, got {act_v:.6e} "
            f"(diff {abs(act_v - exp_v):.3e}, tol {tol:.3e})"
        )


# ---------------------------------------------------------------------------
# Fixture 1 — happy path: small N, modest token count
# ---------------------------------------------------------------------------
def test_happy_path_8_experts_100_tokens():
    torch.manual_seed(0); np.random.seed(0)
    num_experts, n_tokens, top_k, d_hid = 8, 100, 2, 16

    per_expert: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    rng = np.random.default_rng(0)
    for e in range(num_experts):
        active = np.sort(rng.choice(n_tokens, size=max(2, n_tokens * top_k // num_experts),
                                    replace=False))
        idx = torch.from_numpy(active.astype(np.int64))
        g = torch.randn(len(active), d_hid, dtype=torch.float32)
        per_expert[e] = (idx, g)

    expected = _numpy_reference_gated_output_sim(
        {e: (idx.numpy(), g.numpy()) for e, (idx, g) in per_expert.items()}
    )
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    _assert_matches_reference(acc, expected)


# ---------------------------------------------------------------------------
# Fixture 2 — singleton tokens excluded from numerator
# ---------------------------------------------------------------------------
def test_singleton_tokens_excluded():
    """Tokens active in only ONE expert must NOT contribute to any pair sim."""
    num_experts, d_hid = 4, 8
    # Expert 0 active on tokens [0, 1, 2]; expert 1 on [1, 2, 3];
    # expert 2 on [5, 6] (singletons — no overlap with others);
    # expert 3 on [7].
    per_expert = {
        0: (torch.tensor([0, 1, 2], dtype=torch.long),
            torch.randn(3, d_hid, dtype=torch.float32)),
        1: (torch.tensor([1, 2, 3], dtype=torch.long),
            torch.randn(3, d_hid, dtype=torch.float32)),
        2: (torch.tensor([5, 6], dtype=torch.long),
            torch.randn(2, d_hid, dtype=torch.float32)),
        3: (torch.tensor([7], dtype=torch.long),
            torch.randn(1, d_hid, dtype=torch.float32)),
    }
    expected = _numpy_reference_gated_output_sim(
        {e: (idx.numpy(), g.numpy()) for e, (idx, g) in per_expert.items()}
    )
    # Reference must have (0,1) pair (joint on tokens 1, 2) and no others.
    assert set(expected.keys()) == {(0, 1), (1, 0)}, (
        f"reference oracle wrong: {set(expected.keys())}"
    )
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    _assert_matches_reference(acc, expected)


# ---------------------------------------------------------------------------
# Fixture 3 — zero-norm expert: NaN guard
# ---------------------------------------------------------------------------
def test_zero_norm_expert_contributes_zero():
    """An expert whose gated output is all zeros has undefined cosine and must
    contribute 0 (matching the implementation's ``where(isnan, 0)`` guard)."""
    num_experts, d_hid = 3, 8
    per_expert = {
        0: (torch.tensor([0, 1, 2], dtype=torch.long),
            torch.zeros(3, d_hid, dtype=torch.float32)),  # all-zero gated
        1: (torch.tensor([0, 1, 2], dtype=torch.long),
            torch.randn(3, d_hid, dtype=torch.float32)),
        2: (torch.tensor([0, 1, 2], dtype=torch.long),
            torch.randn(3, d_hid, dtype=torch.float32)),
    }
    expected = _numpy_reference_gated_output_sim(
        {e: (idx.numpy(), g.numpy()) for e, (idx, g) in per_expert.items()}
    )
    # Pairs (0, 1) and (0, 2) involve the zero expert → 0 contribution → not
    # in expected dict. Pair (1, 2) is the only one with non-zero sum.
    assert set(expected.keys()) == {(1, 2), (2, 1)}, (
        f"reference oracle wrong: {set(expected.keys())}"
    )
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    _assert_matches_reference(acc, expected)


# ---------------------------------------------------------------------------
# Fixture 4 — no multi-active tokens: early return
# ---------------------------------------------------------------------------
def test_no_multi_active_tokens_early_return():
    """When every token is active in exactly 1 expert, ``gated_output_sim``
    must stay empty."""
    num_experts, d_hid = 4, 8
    per_expert = {
        e: (torch.tensor([e * 10, e * 10 + 1], dtype=torch.long),
            torch.randn(2, d_hid, dtype=torch.float32))
        for e in range(num_experts)
    }
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    # No layer-0 entries should have been written.
    assert all(k[0] != 0 for k in acc.gated_output_sim), (
        f"unexpected gated_output_sim entries: {dict(acc.gated_output_sim)}"
    )


# ---------------------------------------------------------------------------
# Fixture 5 — empty layer: full no-op
# ---------------------------------------------------------------------------
def test_empty_layer_no_op():
    """Layer with no per-expert entries: ``finalize_batch`` must not raise
    and ``gated_output_sim`` must remain empty."""
    num_experts = 4
    acc = ReamCostAccumulator(num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    assert len(acc.gated_output_sim) == 0


# ---------------------------------------------------------------------------
# Fixture 6 — chunk boundary: T_multi just above and below CHUNK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_tokens",
    [_FINALIZE_BATCH_CHUNK - 1, _FINALIZE_BATCH_CHUNK, _FINALIZE_BATCH_CHUNK + 1],
)
def test_chunk_boundary_correctness(n_tokens: int):
    """Crossing the ``_FINALIZE_BATCH_CHUNK`` boundary must not change the
    accumulated sums — chunks are an internal memory optimization."""
    torch.manual_seed(11); np.random.seed(11)
    num_experts, d_hid = 4, 8
    # All 4 experts active on every token → T_multi == n_tokens.
    per_expert = {
        e: (torch.arange(n_tokens, dtype=torch.long),
            torch.randn(n_tokens, d_hid, dtype=torch.float32))
        for e in range(num_experts)
    }
    expected = _numpy_reference_gated_output_sim(
        {e: (idx.numpy(), g.numpy()) for e, (idx, g) in per_expert.items()}
    )
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    # rtol slightly looser at chunk boundaries — different reduction order
    # across chunk sizes may accumulate more rounding.
    _assert_matches_reference(acc, expected, rtol=5e-4)


# ---------------------------------------------------------------------------
# Fixture 7 — Qwen3.6-A3B-shaped: 256 experts, top-k=8, realistic d_hid
# ---------------------------------------------------------------------------
def test_qwen36_a3b_shape():
    """Realistic Qwen3.6-35B-A3B scale: 256 experts, top-k=8, d_hid=2048
    (scaled down from 5120 for test speed)."""
    torch.manual_seed(23); np.random.seed(23)
    num_experts, n_tokens, top_k, d_hid = 256, 512, 8, 2048
    rng = np.random.default_rng(23)
    # Randomly assign top_k experts per token. We materialize the inverse
    # mapping (expert -> sorted token indices) as the actual storage.
    per_expert_lists: dict[int, list[int]] = {e: [] for e in range(num_experts)}
    for t in range(n_tokens):
        chosen = rng.choice(num_experts, size=top_k, replace=False)
        for e in chosen:
            per_expert_lists[int(e)].append(t)

    per_expert: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for e, tokens in per_expert_lists.items():
        if not tokens:
            continue
        idx = torch.tensor(tokens, dtype=torch.long)
        g = torch.randn(len(tokens), d_hid, dtype=torch.float32)
        per_expert[e] = (idx, g)

    expected = _numpy_reference_gated_output_sim(
        {e: (idx.numpy(), g.numpy()) for e, (idx, g) in per_expert.items()}
    )
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    _assert_matches_reference(acc, expected, rtol=5e-4)


# ---------------------------------------------------------------------------
# Fixture 8 — symmetric write contract: (i, j) == (j, i)
# ---------------------------------------------------------------------------
def test_symmetric_write_contract():
    """``finalize_batch`` writes BOTH ``(i, j)`` and ``(j, i)`` with the SAME
    accumulated value."""
    torch.manual_seed(7); np.random.seed(7)
    num_experts, n_tokens, d_hid = 6, 30, 8
    per_expert = {
        e: (torch.arange(n_tokens, dtype=torch.long),
            torch.randn(n_tokens, d_hid, dtype=torch.float32))
        for e in range(num_experts)
    }
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    for (layer, i, j), v in list(acc.gated_output_sim.items()):
        if layer != 0:
            continue
        mirror = acc.gated_output_sim.get((layer, j, i))
        assert mirror is not None, f"missing mirror for ({i}, {j})"
        assert mirror == v, f"({i},{j})={v} but ({j},{i})={mirror}"


# ---------------------------------------------------------------------------
# Fixture 9 — self-pair exclusion
# ---------------------------------------------------------------------------
def test_no_self_pair_entries():
    """``(e, e)`` self-pair entries must never appear in ``gated_output_sim``."""
    torch.manual_seed(13); np.random.seed(13)
    num_experts, n_tokens, d_hid = 5, 30, 8
    per_expert = {
        e: (torch.arange(n_tokens, dtype=torch.long),
            torch.randn(n_tokens, d_hid, dtype=torch.float32))
        for e in range(num_experts)
    }
    acc = _build_acc(per_expert, num_experts=num_experts)
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    for (_layer, i, j) in acc.gated_output_sim:
        assert i != j, f"self-pair leaked: ({i}, {j})"


# ---------------------------------------------------------------------------
# Fixture 10 — _batch_gated_indexed drained on completion
# ---------------------------------------------------------------------------
def test_batch_gated_indexed_drained():
    """After ``finalize_batch``, the input dict for the finalized layer must
    be empty — Stage 2's batch loop relies on this for the next batch."""
    torch.manual_seed(31)
    num_experts, n_tokens, d_hid = 4, 20, 8
    per_expert = {
        e: (torch.arange(n_tokens, dtype=torch.long),
            torch.randn(n_tokens, d_hid, dtype=torch.float32))
        for e in range(num_experts)
    }
    acc = _build_acc(per_expert, num_experts=num_experts)
    assert len(acc._batch_gated_indexed) == num_experts
    acc.finalize_batch(0, num_experts, compute_device=torch.device("cpu"))
    layer_keys = [k for k in acc._batch_gated_indexed if k[0] == 0]
    assert layer_keys == [], f"layer 0 entries leaked: {layer_keys}"
