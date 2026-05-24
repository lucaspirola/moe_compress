"""Stage 2 grouping primitives — leaf helpers shared by cost-matrix and
assignment plugins.

Extracted from ``stage2_reap_ream.py`` in Task 5 of the plugin-architecture
refactor. The three helpers operate on the
``(assignment, centroid_ids, noncentroid_ids)`` triple the assignment solver
produces:

  * ``_apply_skip_merge_floor`` — Direction B percentile-masking of high-cost
    cost-matrix entries to ``+inf``.
  * ``_build_grouped_from_assignment`` — reconstruct
    ``{centroid_id: [centroid_id, *absorbed_member_ids]}`` from a flat
    assignment list (used by EM refinement between rounds).
  * ``_promote_orphans`` — promote any ``centroid_pos < 0`` child to a
    singleton centroid so its weights are not silently dropped.

``stage2_reap_ream`` re-imports all three at module scope.
"""
from __future__ import annotations

import logging

import numpy as np


def _apply_skip_merge_floor(
    delta: np.ndarray, skip_merge_percentile: float,
) -> tuple[np.ndarray, int]:
    """Direction B — skip-merge floor.

    Compute the ``skip_merge_percentile`` percentile ``P`` over the *finite*
    entries of the cost matrix ``delta`` and mask every entry strictly greater
    than ``P`` to ``+inf``. Masked pairs are forbidden to the assignment solver,
    so the affected children fall through the greedy ``-1`` path and become
    singleton ("orphan-promoted") kept experts downstream.

    ``skip_merge_percentile == 100.0`` is the OFF sentinel: the 100th percentile
    equals the maximum finite cost, nothing is strictly above it, so this helper
    returns a fresh copy with no entries masked. (The Stage-2 ``run()`` call site
    skips this helper entirely at the sentinel, leaving the original array as-is.)

    Returns ``(masked_delta, n_masked)`` where ``masked_delta`` is a new array
    (the input is never mutated) and ``n_masked`` is the count of entries newly
    set to ``+inf`` (entries that were already non-finite are not counted).
    """
    out = delta.astype(np.float64, copy=True)
    if out.size == 0:
        return out, 0
    finite_mask = np.isfinite(out)
    if not finite_mask.any():
        # No finite costs at all — percentile is undefined; mask nothing.
        return out, 0
    finite_vals = out[finite_mask]
    p = float(np.percentile(finite_vals, skip_merge_percentile))
    # Strictly above P, and only entries that are currently finite (so we do
    # not "re-mask" already-+inf entries and inflate the reported count).
    above = finite_mask & (out > p)
    n_masked = int(above.sum())
    out[above] = np.inf
    return out, n_masked


def _build_grouped_from_assignment(
    assignment: list[int],
    centroid_ids: list[int],
    noncentroid_ids: list[int],
) -> dict[int, list[int]]:
    """Reconstruct ``{centroid_id: [centroid_id, *absorbed_member_ids]}``
    from a flat assignment list (centroid index per non-centroid)."""
    grouped: dict[int, list[int]] = {c: [c] for c in centroid_ids}
    for child_pos, c_idx in enumerate(assignment):
        if c_idx >= 0:
            grouped[centroid_ids[c_idx]].append(noncentroid_ids[child_pos])
    return grouped


def _promote_orphans(
    grouped: dict[int, list[int]],
    ream_centroid_ids: list[int],
    ream_noncentroid_ids: list[int],
    assignment: list[int],
    *,
    layer_idx: int,
    log: logging.Logger,
) -> None:
    """Promote any unassigned non-centroid (``assignment[child_pos] < 0``) to a
    singleton REAM centroid so its weights are not silently dropped.

    Mutates ``grouped`` and ``ream_centroid_ids`` in place; caller does the
    final ``ream_centroid_ids = sorted(set(...))`` rebind. Emits one WARNING
    per orphan via the injected ``log`` (preserves original ``logger.name``).
    """
    for child_pos, centroid_pos in enumerate(assignment):
        if centroid_pos < 0:
            # Unassigned non-centroid: promote to singleton centroid to avoid weight loss.
            orphan_eid = ream_noncentroid_ids[child_pos]
            log.warning(
                "layer %d: non-centroid expert %d unassigned in capped grouping — "
                "promoted to singleton centroid to avoid weight loss",
                layer_idx, orphan_eid,
            )
            grouped[orphan_eid] = [orphan_eid]
            ream_centroid_ids.append(orphan_eid)
