"""Direction B — skip-merge floor tests (CPU-only, synthetic cost matrices).

Covers the per-layer skip-merge floor implemented in
``stage2_reap_ream._apply_skip_merge_floor`` and consumed in ``run()``:

(a) entries strictly above the percentile become +inf; at/below are untouched
(b) the OFF sentinel (100.0) leaves the cost matrix byte-identical
(c) a masked (high-cost) pair -> that child is left unassigned (-1) by the
    greedy solver -> eligible for orphan promotion
(d) percentile math considers only the *finite* entries of the matrix

These do not require a model, GPU, or any Qwen-specific config — the floor is
pure cost-matrix masking and is therefore model-agnostic by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.stage2.orchestrator import (
    _apply_skip_merge_floor,
    _assign_children_to_centroids,
    _assign_sinkhorn,
)


def test_sinkhorn_orphans_fully_masked_child() -> None:
    """A child whose entire cost row is +inf must become -1 under sinkhorn too,
    so the skip-merge floor's orphan semantics hold for every solver (not just
    greedy/hungarian/mcf)."""
    cost = np.array([
        [0.10, 0.20],          # child 0 — normal
        [np.inf, np.inf],      # child 1 — fully masked by the skip-merge floor
        [0.30, 0.15],          # child 2 — normal
    ], dtype=np.float64)
    assignment = _assign_sinkhorn(cost, n_children=3, n_centroids=2, max_group_cap=2)
    assert assignment[1] == -1, "fully-masked child must orphan under sinkhorn"
    assert assignment[0] >= 0 and assignment[2] >= 0, "unmasked children stay assigned"


# ---------------------------------------------------------------------------
# (a) entries above the percentile -> +inf; entries at/below -> untouched
# ---------------------------------------------------------------------------


def test_above_percentile_masked_below_untouched():
    # 10 finite costs 1..10; P50 of 1..10 is 5.5.
    delta = np.arange(1, 11, dtype=np.float64).reshape(2, 5)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=50.0)

    p = float(np.percentile(delta.ravel(), 50.0))
    assert p == pytest.approx(5.5)

    # Entries strictly above 5.5 (i.e. 6..10) are +inf.
    above = delta > p
    assert np.all(np.isinf(masked[above]))
    assert n_masked == int(above.sum()) == 5

    # Entries at/below 5.5 (1..5) are bit-identical to the input.
    at_or_below = ~above
    np.testing.assert_array_equal(masked[at_or_below], delta[at_or_below])

    # Input matrix is never mutated.
    np.testing.assert_array_equal(delta, np.arange(1, 11).reshape(2, 5))


def test_entry_exactly_at_percentile_is_kept():
    # P100 == max finite cost; the max entry is *at* P, not above -> kept.
    delta = np.array([[1.0, 2.0, 9.0]], dtype=np.float64)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=100.0)
    assert n_masked == 0
    np.testing.assert_array_equal(masked, delta)


# ---------------------------------------------------------------------------
# (b) flag OFF (sentinel 100.0) -> cost matrix byte-identical
# ---------------------------------------------------------------------------


def test_off_sentinel_is_byte_identical():
    rng = np.random.default_rng(0)
    delta = rng.random((7, 4)) * 100.0
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=100.0)
    assert n_masked == 0
    # Byte-identical values (copy is allowed; values must match exactly).
    assert masked.tobytes() == delta.astype(np.float64).tobytes()


def test_off_sentinel_byte_identical_with_some_inf_present():
    # Pre-existing +inf entries must survive the OFF path unchanged and must
    # not be counted as "newly masked".
    delta = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, np.inf]], dtype=np.float64)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=100.0)
    assert n_masked == 0
    np.testing.assert_array_equal(masked, delta)  # NaN-free, so equality holds


# ---------------------------------------------------------------------------
# (c) masked high-cost pair -> child unassigned (-1) by greedy -> orphan-eligible
# ---------------------------------------------------------------------------


def test_masked_child_unassigned_then_orphan_eligible():
    # 3 children, 2 centroids. Child 2 has uniformly huge costs; after the
    # floor those become +inf and child 2 cannot be assigned.
    delta = np.array(
        [
            [0.1, 0.2],   # child 0 — cheap
            [0.3, 0.15],  # child 1 — cheap
            [9.0, 9.5],   # child 2 — expensive, will be masked out
        ],
        dtype=np.float64,
    )
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=60.0)
    assert n_masked == 2  # exactly child 2's two entries masked (deterministic)
    assert np.all(np.isinf(masked[2, :]))
    assert np.all(np.isfinite(masked[:2, :]))

    # Greedy assignment with capacity 2 per centroid.
    assignment = _assign_children_to_centroids(
        masked, n_children=3, n_centroids=2, max_group_cap=2, solver="greedy",
    )
    # Child 2 is unassigned (-1) — this is exactly the value the
    # orphan-promotion loop in run() converts into a singleton centroid.
    assert assignment[2] == -1
    # Children 0 and 1 are still assigned to real centroids.
    assert assignment[0] >= 0
    assert assignment[1] >= 0


def test_off_path_assigns_the_otherwise_masked_child():
    # Same matrix, flag OFF: the expensive child stays finite and *is*
    # assigned — demonstrates the floor is what produces the -1.
    delta = np.array(
        [[0.1, 0.2], [0.3, 0.15], [9.0, 9.5]], dtype=np.float64,
    )
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=100.0)
    assert n_masked == 0
    assignment = _assign_children_to_centroids(
        masked, n_children=3, n_centroids=2, max_group_cap=2, solver="greedy",
    )
    assert assignment[2] >= 0  # no masking -> child 2 gets merged


# ---------------------------------------------------------------------------
# (d) percentile math: only finite entries are considered
# ---------------------------------------------------------------------------


def test_percentile_ignores_inf_entries():
    # Finite costs are 1..5; the rest are +inf. P50 must be the median of
    # 1..5 (== 3.0), NOT polluted by the +inf entries.
    delta = np.array(
        [
            [1.0, 2.0, np.inf],
            [3.0, np.inf, 4.0],
            [5.0, np.inf, np.inf],
        ],
        dtype=np.float64,
    )
    finite_vals = delta[np.isfinite(delta)]
    expected_p = float(np.percentile(finite_vals, 50.0))
    assert expected_p == pytest.approx(3.0)

    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=50.0)
    # Finite entries strictly above 3.0 -> {4.0, 5.0} masked.
    assert n_masked == 2
    assert np.isinf(masked[1, 2])  # 4.0 -> inf
    assert np.isinf(masked[2, 0])  # 5.0 -> inf
    # Finite entries at/below 3.0 untouched.
    assert masked[0, 0] == 1.0
    assert masked[0, 1] == 2.0
    assert masked[1, 0] == 3.0
    # Pre-existing +inf entries are not counted in n_masked and stay +inf.
    assert np.isinf(masked[0, 2])


def test_all_inf_matrix_masks_nothing():
    # Degenerate: no finite entries -> percentile undefined -> mask nothing.
    delta = np.full((3, 3), np.inf, dtype=np.float64)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=10.0)
    assert n_masked == 0
    np.testing.assert_array_equal(masked, delta)


def test_empty_matrix_is_safe():
    delta = np.empty((0, 0), dtype=np.float64)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=50.0)
    assert n_masked == 0
    assert masked.shape == (0, 0)


def test_low_percentile_masks_almost_everything_but_keeps_minimum():
    # P0 == min finite cost; only the strict minimum survives, everything
    # strictly above it is masked.
    delta = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    masked, n_masked = _apply_skip_merge_floor(delta, skip_merge_percentile=0.0)
    assert n_masked == 3
    assert masked[0, 0] == 1.0
    assert np.all(np.isinf(masked[delta > 1.0]))
