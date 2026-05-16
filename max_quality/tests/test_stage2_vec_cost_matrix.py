"""Numerical-equivalence gate for the Stage 2 cost-matrix vectorization.

Stage 2's δ̃_expert(child, centroid) submatrix (REAM Eq. 8) used to be built
by an ``n_nc × n_c`` double loop calling
``ReamCostAccumulator.compute_delta_expert`` once per pair (one lock per
call, ~14.6K calls/layer at Qwen3.6-35B-A3B scale). The vectorized rewrite
reads the dense ``[E, E]`` float64 ``_sim_tensor`` once via
``_extract_sim_expert_matrix_from_tensor`` and broadcasts the rescale.

Stage 2's covariance/merge math decides which experts merge; any drift would
corrupt an in-flight ablation study. This test pins the vectorized path
against an explicit re-implementation of the ORIGINAL double loop at
``atol=1e-9, rtol=1e-7``. CPU-only, synthetic inputs, no model weights.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from moe_compress.stage2_reap_ream import _extract_sim_expert_matrix_from_tensor


# ---------------------------------------------------------------------------
# Reference — the ORIGINAL per-pair path, re-implemented exactly.
# ---------------------------------------------------------------------------
def _reference_compute_delta_expert(
    sim_dict: dict[tuple[int, int], float],
    total_tokens: int,
    expert_i: int,
    expert_j: int,
) -> float:
    """Bit-faithful copy of the pre-vectorization ``compute_delta_expert``.

    Reads the (ordered-pair) gated-output cosine-sum dict and applies the
    REAM Eq. 8 rescale. Returns ``NaN`` when there is no token data.
    """
    sim_val = sim_dict.get((expert_i, expert_j), 0.0)
    if total_tokens == 0:
        return float("nan")
    return float(min(1.0, max(0.0, (sim_val / total_tokens + 1.0) / 2.0)))


def _reference_sim_expert_matrix(
    sim_dict: dict[tuple[int, int], float],
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    total_tokens: int,
) -> np.ndarray:
    """Re-implements the ORIGINAL ``_ream_cost_matrix`` δ̃_expert double loop.

    For each (child, centroid) pair: call the reference
    ``compute_delta_expert``; on NaN substitute 0.5 (the neutral value the
    old loop used). Returns the ``(n_nc, n_c)`` float64 δ̃_expert submatrix
    — i.e. the loop body up to (but not including) the cost combine.
    """
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)
    out = np.zeros((n_nc, n_c), dtype=np.float64)
    for ci in range(n_nc):
        child = noncentroid_ids[ci]
        for cj in range(n_c):
            centroid = centroid_ids[cj]
            sim_expert = _reference_compute_delta_expert(
                sim_dict, total_tokens, child, centroid,
            )
            if math.isnan(sim_expert):
                sim_expert = 0.5  # neutral; matches per-token NaN handling
            out[ci, cj] = sim_expert
    return out


def _dict_to_dense_tensor(
    sim_dict: dict[tuple[int, int], float],
    num_experts: int,
) -> torch.Tensor:
    """Build the dense [E, E] float64 ``_sim_tensor`` from an ordered-pair dict.

    Mirrors how ``finalize_batch`` now stores the accumulator: a symmetric
    matrix with a zero diagonal.
    """
    t = torch.zeros(num_experts, num_experts, dtype=torch.float64)
    for (i, j), v in sim_dict.items():
        t[i, j] = v
    return t


# ---------------------------------------------------------------------------
# Fixtures — synthetic sim dicts of varying shape / sparsity.
# ---------------------------------------------------------------------------
def _make_full_sim_dict(num_experts: int, rng: np.random.Generator) -> dict:
    """Dense symmetric (i,j)/(j,i) cosine-sum dict, zero diagonal."""
    sim: dict[tuple[int, int], float] = {}
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            # Cosine sums can be negative; spread across a realistic range.
            v = float(rng.uniform(-200.0, 200.0))
            sim[(i, j)] = v
            sim[(j, i)] = v
    return sim


def _assert_equiv(
    sim_dict: dict[tuple[int, int], float],
    num_experts: int,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    total_tokens: int,
) -> None:
    """Assert vectorized == reference at atol=1e-9, rtol=1e-7."""
    ref = _reference_sim_expert_matrix(
        sim_dict, noncentroid_ids, centroid_ids, total_tokens,
    )
    sim_tensor = (
        None if not sim_dict and total_tokens == 0
        else _dict_to_dense_tensor(sim_dict, num_experts)
    )
    got = _extract_sim_expert_matrix_from_tensor(
        sim_tensor, noncentroid_ids, centroid_ids, total_tokens,
    )
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert got.dtype == np.float64, f"dtype {got.dtype} != float64"
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-7)


# --- Fixture 1: small (8 experts) ------------------------------------------
def test_small_8_experts():
    rng = np.random.default_rng(0)
    num_experts = 8
    sim_dict = _make_full_sim_dict(num_experts, rng)
    noncentroid_ids = [0, 1, 2, 5]
    centroid_ids = [3, 4, 6, 7]
    _assert_equiv(sim_dict, num_experts, noncentroid_ids, centroid_ids, 1024)


# --- Fixture 2: Qwen-realistic (256 experts, 512 tok, top_k 8) -------------
def test_qwen_realistic_256_experts():
    rng = np.random.default_rng(23)
    num_experts, n_tokens, top_k = 256, 512, 8
    total_tokens = n_tokens

    # Synthesize a finalize_batch-shaped accumulator: cosine sums only for
    # jointly-active pairs, magnitude bounded by min joint-active count.
    per_token_experts = [
        rng.choice(num_experts, size=top_k, replace=False) for _ in range(n_tokens)
    ]
    sim_dict: dict[tuple[int, int], float] = {}
    for chosen in per_token_experts:
        for a_idx in range(len(chosen)):
            for b_idx in range(a_idx + 1, len(chosen)):
                ea, eb = int(chosen[a_idx]), int(chosen[b_idx])
                cos = float(rng.uniform(-1.0, 1.0))
                sim_dict[(ea, eb)] = sim_dict.get((ea, eb), 0.0) + cos
                sim_dict[(eb, ea)] = sim_dict.get((eb, ea), 0.0) + cos

    # Split experts into a centroid / non-centroid partition.
    perm = rng.permutation(num_experts)
    centroid_ids = sorted(int(x) for x in perm[:64])
    noncentroid_ids = sorted(int(x) for x in perm[64:])
    _assert_equiv(sim_dict, num_experts, noncentroid_ids, centroid_ids, total_tokens)


# --- Fixture 3: sparse (~50% zero pairs) -----------------------------------
def test_sparse_half_zero_pairs():
    rng = np.random.default_rng(7)
    num_experts = 32
    sim_dict: dict[tuple[int, int], float] = {}
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            if rng.random() < 0.5:
                continue  # leave this pair absent (zero) — ~50% sparsity
            v = float(rng.uniform(-50.0, 50.0))
            sim_dict[(i, j)] = v
            sim_dict[(j, i)] = v
    noncentroid_ids = list(range(0, num_experts, 2))
    centroid_ids = list(range(1, num_experts, 2))
    _assert_equiv(sim_dict, num_experts, noncentroid_ids, centroid_ids, 4096)


# --- Fixture 4: total=0 (all-NaN → 0.5) ------------------------------------
def test_total_tokens_zero_all_neutral():
    """total_tokens == 0 → compute_delta_expert returns NaN for every pair →
    old loop substitutes 0.5. Vectorized path must return a full-0.5 matrix."""
    num_experts = 16
    noncentroid_ids = [0, 1, 2, 3, 4]
    centroid_ids = [10, 11, 12]
    # Empty sim dict + total=0 → sim_tensor is None in the vectorized path.
    _assert_equiv({}, num_experts, noncentroid_ids, centroid_ids, 0)

    # Also exercise the explicit-None path directly.
    got = _extract_sim_expert_matrix_from_tensor(
        None, noncentroid_ids, centroid_ids, 0,
    )
    assert got.shape == (5, 3)
    np.testing.assert_array_equal(got, np.full((5, 3), 0.5))


def test_sim_tensor_none_with_nonzero_total():
    """A None sim tensor (no batch finalized) with total_tokens > 0 means
    sim_val == 0 for every pair → old algebra yields (0/total+1)/2 == 0.5.
    The vectorized path collapses this to the same full-0.5 matrix."""
    noncentroid_ids = [0, 1, 2]
    centroid_ids = [3, 4]
    got = _extract_sim_expert_matrix_from_tensor(
        None, noncentroid_ids, centroid_ids, 2048,
    )
    np.testing.assert_array_equal(got, np.full((3, 2), 0.5))


# --- Fixture 5: full-overlap (every pair jointly active) -------------------
def test_full_overlap_dense():
    rng = np.random.default_rng(99)
    num_experts = 24
    sim_dict = _make_full_sim_dict(num_experts, rng)
    # Non-centroids and centroids partition all experts, every pair present.
    noncentroid_ids = list(range(0, 12))
    centroid_ids = list(range(12, 24))
    _assert_equiv(sim_dict, num_experts, noncentroid_ids, centroid_ids, 8192)


# --- Extra: clamping — extreme sums push the rescale outside [0, 1] --------
def test_clamp_extreme_values():
    """Huge cosine sums make (sim_val/total + 1)/2 exceed [0, 1]; both the
    reference and the vectorized path must clamp identically."""
    num_experts = 6
    # total small relative to sums → ratios far outside [-1, 1].
    sim_dict = {
        (0, 3): 1e6, (3, 0): 1e6,
        (1, 4): -1e6, (4, 1): -1e6,
        (2, 5): 0.0, (5, 2): 0.0,
    }
    noncentroid_ids = [0, 1, 2]
    centroid_ids = [3, 4, 5]
    _assert_equiv(sim_dict, num_experts, noncentroid_ids, centroid_ids, 10)


# --- Extra: single-element rows / cols -------------------------------------
def test_single_row_single_col():
    rng = np.random.default_rng(5)
    num_experts = 10
    sim_dict = _make_full_sim_dict(num_experts, rng)
    _assert_equiv(sim_dict, num_experts, [4], [7], 512)
    _assert_equiv(sim_dict, num_experts, [0, 1, 2], [9], 512)
    _assert_equiv(sim_dict, num_experts, [3], [5, 6, 8], 512)
