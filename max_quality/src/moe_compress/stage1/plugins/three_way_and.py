"""Phase C₁ — Three-way AND criterion detector (paper Eq. 6).

Paper: arXiv:2507.23279 Eq. 6 (Super-Expert three-way AND criterion).
Migrated from the legacy Stage 1 module in sub-task 8 of the Stage 1 →
plugin-architecture refactor. The plugin dismantles the inline
``_compute_se_thresholds`` + ``_apply_paper_criterion`` helpers and
the three-way-AND branch of ``_collect_candidates``.

The plugin owns three responsibilities:

1. **Statistics computation** — computes ``p995`` (99.5th percentile of
   ``per_expert_max`` values over ``l ∈ L``) and ``a_max`` (max of the
   same) inside :meth:`run`. Writes both, plus ``a_max_threshold =
   a_max_fraction * a_max``, to the context for the orchestrator-built
   ``blacklist_config`` block.
2. **Three-way AND criterion** — for every (l, e) with ``l ∈ L`` and
   ``per_expert_max[(l, e)] > p995`` and ``per_expert_max[(l, e)] >
   a_max_threshold``, add (l, e) to the shared ``CandidateBag`` with
   tag ``"phase_c"``.
3. **No artifact contribution** — the three-way AND statistics live
   inside the orchestrator-built ``blacklist_config`` block (which
   reads them from ctx). The plugin's :meth:`contribute_artifact`
   returns ``{}``.

The plugin's ``writes`` tuple is ``("p995", "a_max", "a_max_threshold",
"candidate_bag")``. The candidate bag is shared with the three other
Phase-C detectors (AIMER / sink-token / magnitude top-K) — see
``tasks/refactor_stage1/subtask_8_plan.md`` §2.2 for the shared-context
pattern.
"""

from __future__ import annotations

import logging

import numpy as np

from .._framework.candidates import CandidateBag
from ..context import Stage1Context

log = logging.getLogger(__name__)


class ThreeWayAndPlugin:
    """Three-way AND criterion detector (Phase C₁, paper Eq. 6).

    Mandatory paper criterion — :meth:`is_enabled` returns ``True``
    unconditionally. Per the overarching plan: "Three-way AND has no
    flag (mandatory paper criterion)".

    The plugin reads ``max_acc`` + ``L`` + ``candidate_bag`` + ``config``
    from ctx; writes ``p995`` + ``a_max`` + ``a_max_threshold`` + mutates
    ``candidate_bag`` in place by adding tagged candidates with tag
    ``"phase_c"``. The orchestrator-built ``blacklist_config`` block
    reads the three statistics from ctx.

    No artifact fragment is contributed (:meth:`contribute_artifact`
    returns ``{}``). The three-way AND statistics live inside the
    ``config`` (``blacklist_config``) block of ``stage1_blacklist.json``,
    not under their own top-level key — the 7-top-level-keys schema
    invariant is preserved.
    """

    name: str = "three_way_and"
    paper: str = "Super-Expert three-way AND criterion (arXiv:2507.23279 Eq. 6)"
    config_key: str = "stage1_grape.super_expert_detection"  # mandatory; no flag
    reads: tuple[str, ...] = (
        "max_acc",
        "L",
        "candidate_bag",
        "config",
    )
    writes: tuple[str, ...] = (
        "p995",
        "a_max",
        "a_max_threshold",
        "candidate_bag",
    )
    provides: tuple[str, ...] = ("downproj_max",)

    def is_enabled(self, config: dict) -> bool:
        """Mandatory paper criterion — always ``True``.

        The three-way AND detector is the load-bearing Super-Expert
        criterion (paper Eq. 6); the overarching plan calls it the
        "mandatory paper criterion" and explicitly says it has no flag.
        Sub-task 10's orchestrator never gates this plugin.
        """
        return True

    def run(self, ctx: Stage1Context) -> None:
        """Compute (p995, a_max) statistics, write three slots, add candidates.

        Reads ``max_acc``, ``L``, ``candidate_bag``, ``config`` from
        ``ctx``. Writes ``p995``, ``a_max``, ``a_max_threshold`` to ctx
        (consumed downstream by ``AimerDetectorPlugin.run`` for its
        layer-max gating, and by the orchestrator for the
        ``blacklist_config`` block). Mutates ``candidate_bag`` in place
        by adding each (l, e) satisfying the three-way AND criterion
        with tag ``"phase_c"``.

        Empty-L semantics: if ``L`` is empty, ``_compute_se_thresholds``
        returns ``(0.0, 0.0)``, ``a_max_threshold = 0.0``, and the
        per-(l, e) loop short-circuits because ``per_expert_max[(l, e)]
        > 0.0`` cannot be satisfied for any ``l ∈ L = ∅``. The three
        statistics are still written to ctx so the ``blacklist_config``
        block reads valid floats.
        """
        max_acc = ctx.get("max_acc")
        L: set[int] = ctx.get("L")
        candidate_bag: CandidateBag = ctx.get("candidate_bag")
        config: dict = ctx.get("config")

        per_expert_max: dict[tuple[int, int], float] = max_acc.per_expert_max

        s1 = config["stage1_grape"]
        se_cfg = s1.get("super_expert_detection", {})
        # a_max_fraction lives at config["stage1_grape"]["super_expert_detection"]
        # ["a_max_fraction"] — confirmed against the legacy Stage 1 module. Default 0.1.
        a_max_fraction = float(se_cfg.get("a_max_fraction", 0.1))

        p995, a_max = _compute_se_thresholds(per_expert_max, L)
        a_max_threshold = a_max_fraction * a_max

        log.info(
            "Stage 1 Phase C₁ (three-way AND): P99.5=%.3g, a_max=%.3g, "
            "a_max_threshold=%.3g, |L|=%d.",
            p995, a_max, a_max_threshold, len(L),
        )

        per_layer_paper = _apply_paper_criterion(per_expert_max, L, p995, a_max_threshold)
        for li, exps in per_layer_paper.items():
            for e in exps:
                candidate_bag.add(int(li), int(e), "phase_c")

        ctx.set("p995", float(p995))
        ctx.set("a_max", float(a_max))
        ctx.set("a_max_threshold", float(a_max_threshold))

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
        """Return ``{}`` — three-way AND statistics live in ``blacklist_config``.

        The orchestrator reads ``p995`` / ``a_max`` / ``a_max_threshold``
        from ctx and emits them under the ``config`` top-level key
        (which is orchestrator-owned, not plugin-owned). The plugin
        contributes no top-level fragment. Empty-dict return is
        Protocol-compliant.
        """
        return {}


# ---------------------------------------------------------------------------
# Phase C₁ private helpers — moved verbatim from the legacy Stage 1 module
# in sub-task 8. Sole caller: :class:`ThreeWayAndPlugin.run`.
# ---------------------------------------------------------------------------


def _compute_se_thresholds(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
) -> tuple[float, float]:
    """Compute P99.5 and a_max over all (l, e) with l ∈ L.

    Moved verbatim from the legacy Stage 1 module in sub-task 8.
    Sole caller: :class:`ThreeWayAndPlugin.run`.
    """
    A = [v for (li, _e), v in per_expert_max.items() if li in L]
    if not A:
        if L:
            log.warning(
                "_compute_se_thresholds: MA-formation layers L=%s but no expert fired "
                "on any calibration sample in those layers; SE detection will find nothing. "
                "Consider increasing the calibration set size or checking the model.",
                sorted(L),
            )
        return 0.0, 0.0
    arr = np.array(A, dtype=np.float64)
    p995 = float(np.percentile(arr, 99.5))
    a_max = float(arr.max())
    return p995, a_max


def _apply_paper_criterion(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    p995: float,
    a_max_threshold: float,
) -> dict[int, list[int]]:
    """Apply Eq. 6 three-way AND: a > P99.5 AND a > 0.1·a_max AND l ∈ L.

    Moved verbatim from the legacy Stage 1 module in sub-task 8.
    Returns ``{layer_idx: [expert_idx, ...]}``; per-layer expert lists
    appear in insertion order (the order they iterate from
    ``per_expert_max.items()``).
    """
    if not L:
        if per_expert_max:
            log.warning(
                "Phase C: L is empty; skipping SE detection (no MA-formation layers found, "
                "even after fallback)."
            )
        return {}
    # Magnitudes are collected for ALL MoE layers (spec §4 Phase B: "All MoE layers
    # are instrumented simultaneously"); the SE three-way AND is then enforced here
    # by silently skipping any (l, e) with l ∉ L (Eq. 6's `l ∈ L` clause).
    blacklist: dict[int, list[int]] = {}
    for (li, e), v in per_expert_max.items():
        if li not in L:
            continue
        if v > p995 and v > a_max_threshold:
            blacklist.setdefault(li, []).append(e)
    return blacklist
