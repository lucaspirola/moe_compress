"""Magnitude top-K Super-Expert candidate detector (project-original).

Paper
-----
**None — there is no paper for this detector.** It is project-original.
arXiv:2507.23279 (the SE paper this stage otherwise derives from)
prescribes the three-way AND criterion (see ``three_way_and.py``) as the
sole SE-detection rule and contains no top-K-by-magnitude heuristic.

The GRAPE proposal (arXiv:2604.06542) is occasionally near this file
because Stage 1 lives in the ``grape`` config block, but it is unrelated
to expert ranking by per-layer magnitude.

Official code
-------------
None. The companion paper's official repo
(ZunhaiSu/Super-Experts-Profilling @
``573aead3127ae593ba267758b832944f8fed1485``) implements only the
three-way AND criterion. This detector has no upstream reference
implementation.

Deviation: D-magnitude-topk-candidates
--------------------------------------
Project-original candidate-pool extension. For each MA-formation layer
``l ∈ L``, augment the candidate pool with the top-``magnitude_topk_per_l_layer``
experts ranked by ``per_expert_max[(l, e)]`` (down-projection-output
absolute-max accumulated over the calibration pass), regardless of
whether they pass the three-way AND threshold. Provenance tagged
``"magnitude_topk"``.

The final blacklist is gated by the downstream ablation-filter pass —
magnitude-top-K candidacy is necessary, not sufficient. False positives
cost ablation walltime but cannot reach the final blacklist without
measurable per-candidate ΔNLL.

Why this detector exists
------------------------
The static three-way AND threshold from arXiv:2507.23279 is tuned for
the paper's measured models (e.g. Qwen3-30B-A3B initial release /
DeepSeek-V2-Lite / DeepSeek-R1 / Mixtral-8x7B; see Table 1); on
architecture-shifted models the right thresholds shift too. The v3 Phase F audit (commits in the 2026-05-10 series) surfaced
non-blacklisted experts with measurable ablation ΔNLL — e.g.,
``L34E85`` with ΔNLL ≈ −0.025 — that the static three-way AND missed.
A magnitude-top-K source per layer catches these by widening the
candidate pool without weakening the final filter (which remains
ablation-gated).

K = 16 is set to **2× the model's active-experts-per-token** (top-8
routing on the target Qwen3-30B-A3B-2507 architecture). The 2× factor is
broad enough to recover SEs that fell just below the three-way AND's
full criterion (``a_{l,e} > P99.5 ∧ a_{l,e} > a_max/10 ∧ l ∈ L``;
arXiv:2507.23279 Eq. 6) while still bounding the per-layer ablation cost.

Git archaeology
---------------
- ``3ad418b``/``3c48d76`` (2026-05-10) "feat(stage1): magnitude top-K +
  tightened sink-token candidate selection" — initial extraction of
  ``_magnitude_topk_candidates(per_expert_max, L, top_k)`` from
  ``_collect_candidates`` into a named helper.
- ``a2e34db``/``aa2ed94`` (same v6 wave as sink-token tightening) —
  ``magnitude_topk_per_l_layer: 16`` added as the production-config
  default. Same commit raised the sink-token thresholds; both are part
  of the v6 "candidate-pool + ablation-filter" architecture switch.
- ``94b5526``/``9bdbda8`` "spec(stage1): §12 rewrites
  — candidates+ablation-filter for AIMER/sink/topk" — recorded the
  ``L34E85 ΔNLL ≈ −0.025`` motivating anecdote.

Naming-history note
-------------------
The legacy stage-1 monolith called this "Phase C₄" (fourth sub-source of
the unified Phase C candidate-collection stage). The current plugin
architecture has no phase taxonomy — plugin enable/disable is the flag
that gates execution. Only the legacy log string ``"Stage 1 Phase C₄"``
is retained for dashboard back-compat; the provenance tag attached to
candidates produced by this plugin is ``"magnitude_topk"`` (see line
where ``candidate_bag.add(..., "magnitude_topk")`` is called), not
``"phase_c"``. New prose drops the legacy labels.

Plugin contract
---------------
``writes = ("candidate_bag",)`` — only the shared bag is mutated, via
``add(l, e, "magnitude_topk")``. ``provides = ("downproj_max",)`` is the
declarative metadata advertising that this detector reads from the
``per_expert_max`` accumulator (shared with the three-way AND plugin —
the calibration pass collects it once for both consumers).

``contribute_artifact`` returns ``{}`` — the magnitude-top-K parameter
(``magnitude_topk_per_l_layer``) lives inside the orchestrator-built
``config`` (``blacklist_config``) top-level block of
``stage1_blacklist.json``; this plugin contributes no top-level fragment.
The candidates' per-layer breakdown is emitted by the shared candidate
bag's ``to_provenance_dict()`` under ``magnitude_topk.candidates``.

``is_enabled`` gates on ``magnitude_topk_per_l_layer > 0``; the value
``0`` disables the detector entirely (also short-circuited inside the
helper as ``top_k <= 0``).
"""

from __future__ import annotations

import logging

from ...pipeline.candidates import CandidateBag
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class MagnitudeTopkPlugin:
    """Magnitude top-K Super-Expert candidate detector (project-original).

    For each ``l ∈ L``, picks the top-``magnitude_topk_per_l_layer``
    experts by ``per_expert_max[(l, e)]`` and adds each to the shared
    ``CandidateBag`` with tag ``"magnitude_topk"``. There is no paper
    for this detector — see deviation D-magnitude-topk-candidates in
    the module docstring for the rationale, the K=16 = 2×top-routing
    derivation, and the v3 Phase F motivating evidence.

    Gated by ``magnitude_topk_per_l_layer > 0`` via :meth:`is_enabled`.
    Default K is 16. Final blacklist requires ablation-filter evidence.

    No artifact fragment is contributed (:meth:`contribute_artifact`
    returns ``{}``). The 7-top-level-keys schema invariant is
    preserved — the magnitude-topk parameter lives inside the ``config``
    (``blacklist_config``) block.
    """

    name: str = "magnitude_topk"
    paper: str = (
        "Magnitude top-K candidate source (project-original; no paper). "
        "arXiv:2507.23279 prescribes only the three-way AND criterion — "
        "official code ZunhaiSu/Super-Experts-Profilling @ "
        "573aead3127ae593ba267758b832944f8fed1485 implements no top-K-by-"
        "magnitude heuristic. See deviation D-magnitude-topk-candidates "
        "in the module docstring."
    )
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

        Default value 16 (per the v6 production-config default). When
        ``0`` (or any value ≤ 0), the plugin is disabled. The orchestrator
        gates the plugin on this; the run also short-circuits on the
        top_k=0 check inside :func:`_magnitude_topk_candidates`.
        """
        s1 = config.get("stage1_grape", {})
        se = s1.get("super_expert_detection", {})
        return int(se.get("magnitude_topk_per_l_layer", 16)) > 0

    def run(self, ctx: PipelineContext) -> None:
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

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return ``{}`` — magnitude-topk parameter lives in ``blacklist_config``.

        The orchestrator reads ``magnitude_topk_per_l_layer`` directly
        from the config and emits it under the ``config`` top-level
        key. The plugin contributes no top-level fragment. Empty-dict
        return is Protocol-compliant.
        """
        return {}


# ---------------------------------------------------------------------------
# Private helper — moved verbatim from the legacy Stage 1 module in
# sub-task 8. Sole caller: :class:`MagnitudeTopkPlugin.run`.
# ---------------------------------------------------------------------------


def _magnitude_topk_candidates(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    top_k: int,
) -> set[tuple[int, int]]:
    """For each ``l ∈ L``, pick the top-``top_k`` experts by ``per_expert_max``.

    Catches large-magnitude experts within MA-formation layers that don't
    quite cross the three-way AND but should still go through the
    ablation filter. See deviation D-magnitude-topk-candidates in the
    module docstring for the rationale and the v3 Phase F motivating
    evidence.

    Moved verbatim from the legacy Stage 1 module in sub-task 8. Sole
    caller: :class:`MagnitudeTopkPlugin.run`.

    Implementation note
    -------------------
    Ties on ``per_expert_max[(l, e)]`` are broken by **ascending expert
    index** via the secondary sort key ``(-v, e)``. This makes the
    selection deterministic across Python/CPython releases (which is
    otherwise dependent on ``dict``-iteration order of
    ``per_expert_max``) and reproducible across runs.
    """
    if top_k <= 0 or not L:
        return set()
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (li, e), v in per_expert_max.items():
        if li in L:
            by_layer.setdefault(int(li), []).append((int(e), float(v)))
    out: set[tuple[int, int]] = set()
    for li, lst in by_layer.items():
        # Primary: descending magnitude. Secondary: ascending expert id
        # for deterministic tie-break (see Implementation note above).
        lst.sort(key=lambda t: (-t[1], t[0]))
        for e, _ in lst[:top_k]:
            out.add((li, e))
    return out
