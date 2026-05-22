"""Phase C₄ — Magnitude top-K detector.

Catches large-magnitude experts within MA-formation layers (``l ∈ L``)
that don't quite cross the three-way AND criterion but should still
be ablation-tested. See `ALGORITHM_REFERENCE.md` §12 [D-magnitude-topk-candidates].

Migrated from the legacy Stage 1 module in sub-task 8 of the Stage 1 →
plugin-architecture refactor. The plugin dismantles the inline
``_magnitude_topk_candidates`` helper and the magnitude-topk branch
of ``_collect_candidates``.

The plugin owns two responsibilities:

1. **Top-K selection** — for each ``l ∈ L``, pick the top-K experts by
   ``per_expert_max[(l, e)]`` and add each to the shared
   ``CandidateBag`` with tag ``"magnitude_topk"``.
2. **No artifact contribution** — the magnitude-topk parameter
   (``magnitude_topk_per_l_layer``) lives inside the orchestrator-built
   ``blacklist_config`` block. The plugin's :meth:`contribute_artifact`
   returns ``{}``.

The plugin's ``writes`` tuple is ``("candidate_bag",)`` — only the
shared bag is mutated. The plugin is gated by
``magnitude_topk_per_l_layer > 0`` (default 16; the typical disable
value is 0). See ``tasks/refactor_stage1/subtask_8_plan.md`` §2.10 for
the is_enabled contract.
"""

from __future__ import annotations

import logging

from .._framework.candidates import CandidateBag
from ..context import Stage1Context

log = logging.getLogger(__name__)


class MagnitudeTopkPlugin:
    """Magnitude top-K detector (Phase C₄).

    For each ``l ∈ L``, picks the top-``magnitude_topk_per_l_layer``
    experts by ``per_expert_max[(l, e)]`` and adds each to the shared
    ``CandidateBag`` with tag ``"magnitude_topk"``.

    Gated by ``magnitude_topk_per_l_layer > 0`` via :meth:`is_enabled`.
    Default top-K is 16 (matching the legacy default). The orchestrator
    gates the plugin on :meth:`is_enabled`; :meth:`run` also short-circuits
    on the top_k=0 check inside :func:`_magnitude_topk_candidates`.

    No artifact fragment is contributed (:meth:`contribute_artifact`
    returns ``{}``). The 7-top-level-keys schema invariant is
    preserved — the magnitude-topk parameter lives inside the ``config``
    (``blacklist_config``) block.
    """

    name: str = "magnitude_topk"
    paper: str = "Magnitude top-K per-layer Super-Expert candidate (D-magnitude-topk-candidates)"
    config_key: str = "stage1_grape.super_expert_detection.magnitude_topk_per_l_layer"
    reads: tuple[str, ...] = (
        "max_acc",
        "L",
        "candidate_bag",
        "config",
    )
    writes: tuple[str, ...] = (
        "candidate_bag",
    )
    provides: tuple[str, ...] = ("downproj_max",)

    def is_enabled(self, config: dict) -> bool:
        """Read ``config["stage1_grape"]["super_expert_detection"]
        ["magnitude_topk_per_l_layer"]``; return ``True`` iff the value > 0.

        Default value 16 (per the legacy default). When ``0`` (or
        any value ≤ 0), the plugin is disabled. The orchestrator gates
        the plugin on this; the run also short-circuits on
        the top_k=0 check.
        """
        s1 = config.get("stage1_grape", {})
        se = s1.get("super_expert_detection", {})
        return int(se.get("magnitude_topk_per_l_layer", 16)) > 0

    def run(self, ctx: Stage1Context) -> None:
        """Add top-K magnitude candidates per ``l ∈ L`` to the shared bag.

        Reads ``max_acc``, ``L``, ``candidate_bag``, ``config`` from
        ``ctx``. Mutates ``candidate_bag`` in place by adding each (l, e)
        in the top-K set with tag ``"magnitude_topk"``. Short-circuits
        on ``top_k <= 0`` or empty ``L`` (see
        :func:`_magnitude_topk_candidates`).

        Iteration order: legacy ``_magnitude_topk_candidates`` returns
        a ``set[tuple[int, int]]`` with no guaranteed iteration order.
        ``CandidateBag.add(l, e, tag)`` uses the bag's internal dict
        insertion order, which depends on the set's iteration order;
        however the bag's ``to_provenance_dict`` is finally consumed by
        the orchestrator either as ``(l, e) -> sorted(tags)`` (the
        sort is per-pair, not cross-pair) or as a tag-keyed inversion
        (downstream sorts experts per layer). The golden-snapshot
        invariant is therefore unaffected by the within-set iteration
        order — what matters is the final per-key tag sort and the
        per-tag-inversion's sorted-experts list. **Both are sorted at
        emission time**, so the bag's insertion order of magnitude-topk
        (l, e) pairs does not propagate to the artifact.
        """
        max_acc = ctx.get("max_acc")
        L: set[int] = ctx.get("L")
        candidate_bag: CandidateBag = ctx.get("candidate_bag")
        config: dict = ctx.get("config")

        per_expert_max: dict[tuple[int, int], float] = max_acc.per_expert_max
        s1 = config["stage1_grape"]
        se_cfg = s1.get("super_expert_detection", {})
        top_k = int(se_cfg.get("magnitude_topk_per_l_layer", 16))

        log.info(
            "Stage 1 Phase C₄ (magnitude top-K): top_k=%d, |L|=%d.",
            top_k, len(L),
        )

        pairs = _magnitude_topk_candidates(per_expert_max, L, top_k)
        for (li, e) in pairs:
            candidate_bag.add(int(li), int(e), "magnitude_topk")

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
        """Return ``{}`` — magnitude-topk parameter lives in ``blacklist_config``.

        The orchestrator reads ``magnitude_topk_per_l_layer`` directly
        from the config and emits it under the ``config`` top-level
        key. The plugin contributes no top-level fragment. Empty-dict
        return is Protocol-compliant.
        """
        return {}


# ---------------------------------------------------------------------------
# Phase C₄ private helper — moved verbatim from the legacy Stage 1 module
# in sub-task 8. Sole caller: :class:`MagnitudeTopkPlugin.run`.
# ---------------------------------------------------------------------------


def _magnitude_topk_candidates(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    top_k: int,
) -> set[tuple[int, int]]:
    """For each ``l ∈ L``, pick the top-``top_k`` experts by ``per_expert_max``.

    Catches large-magnitude experts within MA-formation layers that don't
    quite cross the three-way AND but should still go through Phase D
    ablation. See [D-magnitude-topk-candidates] in ALGORITHM_REFERENCE.md §12.

    Moved verbatim from the legacy Stage 1 module in sub-task 8. Sole
    caller: :class:`MagnitudeTopkPlugin.run`.
    """
    if top_k <= 0 or not L:
        return set()
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (li, e), v in per_expert_max.items():
        if li in L:
            by_layer.setdefault(int(li), []).append((int(e), float(v)))
    out: set[tuple[int, int]] = set()
    for li, lst in by_layer.items():
        lst.sort(key=lambda t: -t[1])
        for e, _ in lst[:top_k]:
            out.add((li, e))
    return out
