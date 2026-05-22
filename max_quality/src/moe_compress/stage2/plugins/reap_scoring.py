"""REAP scoring plugin: owns the per-layer ReapAccumulator and derives
the ``scores`` / ``freq`` slots so downstream phases (centroid selection,
cost-matrix construction) can read them as plain slots on the per-layer
:class:`PipelineContext`.

This plugin is the first real extraction out of ``LegacyAdapter`` (T7 of the
plugin refactor). The pure-function ``select_centroids_by_reap`` packages the
centroid-selection inner loop so it can be unit-tested without spinning up a
full pipeline.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from ...pipeline.context import PipelineContext
from ...utils.activation_hooks import ReapAccumulator

log = logging.getLogger(__name__)


class ReapScoringPlugin:
    """Construct and finalize the layer's ReapAccumulator; publish scores/freq."""

    name = "reap_scoring"
    paper = "REAP scoring: per-layer ReapAccumulator deriving scores/freq slots."
    config_key = "stage2_reap_ream"
    reads: tuple[str, ...] = ("layer_ref",)
    writes: tuple[str, ...] = ("reap_acc", "scores", "freq")
    provides: tuple[str, ...] = ("reap_acc",)

    def is_enabled(self, config: dict) -> bool:
        """Always-on; REAP scoring is mandatory for REAM centroid choice."""
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Phase: on_layer_setup (runs BEFORE LegacyAdapter.on_layer_setup
    # because ReapScoringPlugin is registered first; see stage2_reap_ream.py).
    # ------------------------------------------------------------------
    def on_layer_setup(self, ctx: PipelineContext) -> None:
        """Create the per-layer ReapAccumulator and stash it on ctx."""
        ctx.set("reap_acc", ReapAccumulator())

    # ------------------------------------------------------------------
    # Phase: on_score (NEW in T7; runs between on_profile and compute_assignment)
    # ------------------------------------------------------------------
    def on_score(self, ctx: PipelineContext) -> None:
        """Finalize the accumulator and populate ctx.scores + ctx.freq.

        ``scores`` is an ``np.ndarray`` of length ``n_experts`` containing the
        REAP saliency for each expert (centroid candidacy ordering).
        ``freq`` is a ``dict[int, int]`` mapping expert id to total routed
        token count (used by `min_active_tokens` filtering and by some cost
        matrices).
        """
        layer_ref = ctx.get("layer_ref")
        layer_idx = layer_ref.layer_idx
        n_experts = layer_ref.num_routed_experts

        reap_acc = ctx.get("reap_acc")
        # Single finalize call; mirrors the pre-T7 LegacyAdapter.on_profile slice.
        reap_acc.finalize_layer(layer_idx)

        # Derive the per-expert score vector and the freq dict. Build them as
        # plain numpy + dict so downstream consumers (and unit tests) do not
        # take a hidden dependency on ReapAccumulator's internal layout.
        ctx.set("scores", np.array(
            [reap_acc.score(layer_idx, e) for e in range(n_experts)]
        ))
        ctx.set("freq", {
            e: reap_acc.freq.get((layer_idx, e), 0)
            for e in range(n_experts)
        })


def select_centroids_by_reap(
    scores: np.ndarray,
    freq: dict[int, int],
    *,
    ream_target: int,
    min_active_tokens: int,
    protected: Iterable[int],
    layer_idx: int,
    log: logging.Logger,
) -> list[int]:
    """Pick up to ``ream_target`` REAM centroids by REAP score, descending.

    Spec §5 Step 3 (greedy centroid selection) + Spec D-reap-min-active-tokens
    (§12, low-frequency filter). Protected experts are never centroids — their
    weights pass through Stage 2 unchanged.

    Parameters
    ----------
    scores:
        1-D ``np.ndarray`` of REAP saliencies, length == n_experts.
        Indexed by expert id.
    freq:
        Mapping from expert id to routed-token count.
    ream_target:
        Maximum number of centroids to return. If 0, returns ``[]``.
    min_active_tokens:
        Experts with ``freq[e] < min_active_tokens`` are filtered out of
        centroid candidacy (they become non-centroids and merge via
        Hungarian alignment instead).
    protected:
        Expert ids that are protected (super-experts + shared experts from
        ``stage1_blacklist.json``). Never returned as centroids.
    layer_idx:
        Logged in the under-budget warning so operators can correlate.
    log:
        Logger to emit the under-budget warning through.

    Returns
    -------
    list[int]
        Centroid expert ids, in selection order (highest score first).
        May be shorter than ``ream_target`` when the min-active-tokens
        filter eliminates candidates; the caller is expected to warn or
        bump the target.
    """
    if ream_target <= 0:
        return []
    protected_set = set(protected)
    selected: list[int] = []
    # np.argsort(-scores) gives descending-score iteration order, matching
    # the pre-T7 LegacyAdapter loop verbatim.
    for _e in np.argsort(-scores):
        if len(selected) >= ream_target:
            break
        e = int(_e)
        if e in protected_set:
            continue
        if freq.get(e, 0) < min_active_tokens:
            # Spec D-reap-min-active-tokens (§12): low-frequency experts
            # are filtered from centroid candidacy.
            continue
        selected.append(e)

    if len(selected) < ream_target:
        log.warning(
            "  layer %d: REAM centroid selection yielded %d < %d — "
            "%d candidate(s) filtered by reap_min_active_tokens=%d "
            "(per spec D-reap-min-active-tokens)",
            layer_idx, len(selected), ream_target,
            ream_target - len(selected), min_active_tokens,
        )
    return selected
