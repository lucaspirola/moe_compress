"""Shared per-layer floor helper for Stage 1 plugins.

Centralises the floor-and-k_max derivation consumed by
:mod:`stage1.plugins.grape_merge` and
:mod:`stage1.plugins.damage_curve_dp` so the DP plan stays feasible
against GRAPE's downstream constraints by construction.

Definitions
-----------
For a layer with ``n`` experts of which ``n_blacklisted`` are protected
super-experts (cannot be merged), under floor divisor ``d``:

* ``total_floor`` = ``max(n // d, n_blacklisted)`` — minimum TOTAL
  surviving experts including the blacklist. The ``max`` clamps the
  configured floor up to the blacklist count when the blacklist already
  exceeds the configured floor.
* ``non_bl_floor`` = ``max(n // d − n_blacklisted, 0)`` — the
  non-blacklisted portion of the floor. GRAPE's ``cluster_counts[li]``
  (which excludes blacklisted experts) is forbidden from dropping below
  this value.
* ``k_max`` = ``n − total_floor`` — the maximum number of merge steps a
  layer can absorb without violating its total floor. Equal to the
  cluster-count decrement budget for that layer.

These three quantities satisfy ``total_floor + k_max == n`` and
``non_bl_floor + k_max == n − n_blacklisted == initial cluster_count``
by construction.

Used by GRAPE for ``floors[li]`` (non_bl_floor) and by S1_DP for
``k_max[li]``. Centralising the formula here ensures the DP plans
merges against the same floor GRAPE will enforce downstream — any
future change (e.g. a per-layer floor schedule) only needs editing in
one place.
"""

from __future__ import annotations

from typing import NamedTuple


class _LayerFloor(NamedTuple):
    """Per-layer floor decomposition.

    Attributes
    ----------
    non_bl_floor:
        Floor on the non-blacklisted cluster count (GRAPE's ``floors[li]``).
    total_floor:
        Floor on the TOTAL surviving expert count (blacklist + non-bl).
    k_max:
        Maximum merge-step budget for the layer (= ``n − total_floor``).
    """

    non_bl_floor: int
    total_floor: int
    k_max: int


def per_layer_floor(
    n: int, n_blacklisted: int, floor_divisor: int
) -> _LayerFloor:
    """Return the ``(non_bl_floor, total_floor, k_max)`` triple for a layer.

    See module docstring for the definition of each quantity.

    Parameters
    ----------
    n:
        Total expert count in the layer.
    n_blacklisted:
        Number of blacklisted (protected) experts in the layer.
    floor_divisor:
        Divisor for the configured floor; ``n // floor_divisor`` is the
        raw floor (clamped up to ``n_blacklisted``).

    Raises
    ------
    ValueError
        If ``floor_divisor < 1``.
    """
    if floor_divisor < 1:
        raise ValueError(
            f"per_layer_floor: floor_divisor must be >= 1, got {floor_divisor}"
        )
    total_floor = max(n // floor_divisor, n_blacklisted)
    non_bl_floor = max(n // floor_divisor - n_blacklisted, 0)
    k_max = max(0, n - total_floor)
    return _LayerFloor(
        non_bl_floor=non_bl_floor, total_floor=total_floor, k_max=k_max
    )
