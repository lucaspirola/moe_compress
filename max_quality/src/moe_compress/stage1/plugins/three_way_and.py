"""Super-Expert three-way AND criterion (load-bearing SE detector).

Paper
-----
Su et al., "Super Experts in MoE Models", arXiv:2507.23279 (2025).
**Equation 6** (source.md lines 395-402):

    a_{l,e} > P99.5 AND a_{l,e} > (1/10) · a_max AND l ∈ L     (Eq. 6)

where ``a_{l,e}`` is the maximum output magnitude of expert ``e`` in
layer ``l`` to its ``down_proj``, ``A = {a_{l,e}}`` is the set across
the eligible (layer, expert) pairs, ``P99.5 = Percentile_{99.5}(A)``,
``a_max = max A``, and ``L`` is the set of MA-formation layers
produced by :class:`MADetectionPlugin`. (source.md line 402: *"This
criterion is motivated by the heavy-tailed distribution of a_{l,e}
and effectively identifies the experts of interest across various
MoE LLMs"*.)

Algorithm 1 Stage 2 (source.md lines 1996-2003) renders the same
criterion as the loop body:

    for each (l, e) with a_{l,e} ∈ A:
        if a_{l,e} > P99.5 and a_{l,e} > (1/10)·a_max:
            S ← S ∪ {(l, e)}

The ``l ∈ L`` factor is enforced one nesting level up — Algorithm 1
step 16 (source.md:1981) ``for each layer l ∈ L`` — so it does not
appear inside the inner ``if``. The paper's two renderings (§3.2.1 prose with ``l ∈ L``
inside Eq. 6, and Algorithm 1's pseudocode with ``l ∈ L`` at the
layer-loop header) are mathematically equivalent.

Official implementation (golden reference)
------------------------------------------
``github.com/ZunhaiSu/Super-Experts-Profilling`` pinned to commit
``573aead3127ae593ba267758b832944f8fed1485`` (default branch ``main``
HEAD, dated 2025-09-25). The criterion lives at
``eval_utils.py:619-651`` (``_identify_super_experts``); the exact
filter is line 642:

    if item['output_max'] > percentile and \\
       item['output_max'] > np.max(output_max_values) // times:
        # quantile=99.5, times=10 (defaults from the signature)
        Super_Experts.append(...)

The ``// times`` integer-floor-divide with ``times=10`` is the
official code's encoding of the paper's ``a_max / 10`` threshold.
Note: the official function operates on the **union**
``output_max_values`` over all layers — the ``l ∈ L`` restriction
is NOT enforced inside ``_identify_super_experts``. The official
pipeline instead applies an unrelated ``include_layers=0.75`` depth
heuristic in ``_super_experts_analysis``
(``eval_utils.py:470-479``); that heuristic filters by absolute
layer index (``layer_index < round(total_layers * 0.75)``), not by
``L``. The official ``L``-equivalent (the MA-formation layer set)
does not exist as a code object — only as Algorithm 1 pseudocode.

Deviations from paper
---------------------
**D-SE-A — ``A`` restricted to ``l ∈ L``** (resolves the prose-vs-
pseudocode ambiguity in the paper).

* Paper §3.2.1 prose: *"all such values across the entire model"*
  (source.md line 392-393) — implies ``A`` is the union over every
  layer.
* Paper Algorithm 1 Stage 2 block (lines 1980-1992 of source.md):
  the inner ``a_{l,e} ∈ A`` loop is wrapped by
  ``for each layer l ∈ L`` (source.md line 1981) — meaning the
  outer A-construction loop also runs only on ``l ∈ L``, so ``A``
  is the layer-restricted set.

Implementation follows Algorithm 1 (the procedurally precise
rendering). The §3.2.1 prose is imprecise and contradicted by the
pseudocode. Note: the authors' official code does NOT validate
either reading — it uses a depth-fraction heuristic
(``include_layers=0.75``: keep layers with
``layer_index < round(total_layers * 0.75)``; see ``run.py:28``
declaring the CLI arg and ``eval_utils.py:470-479`` consuming it),
which is unrelated to the MA-formation layer set ``L`` from Stage 1.
The plugin therefore implements the paper's Algorithm 1 directly,
without an official-code analogue for the ``l ∈ L`` clause.

**D-a-max-fraction — ``a_max_fraction`` is a configurable knob**.

* Paper Eq. 6 fixes the multiplier at exactly ``1/10`` (= 0.1).
* Implementation exposes it as
  ``stage1_grape.super_expert_detection.a_max_fraction`` with
  default ``0.1`` (matches the paper). Production runs MUST keep
  ``a_max_fraction=0.1`` for paper-compliant SE detection.

The knob exists so an operator can sweep ablations on the second
SE-criterion threshold without code changes; it is not a quality
improvement over the paper's fixed value.

**D-amax-mult-real-vs-floor — real multiplication, not floor-divide**.

* Paper Algorithm 1 step 28 (source.md:1997-1999) uses real division:
  ``a_{l,e} > (1/10)·a_max``.
* Official code (``eval_utils.py:642``) uses NumPy floor-divide
  (``//``) on a float scalar: ``np.max(output_max_values) // 10``,
  which returns ``floor(a_max / 10)`` — looser than the paper's real
  ``a_max/10`` whenever ``a_max`` is not a multiple of 10
  (e.g., ``a_max=99.7`` → paper threshold 9.97, official threshold
  ``np.float64(9.0)``; the official threshold is *looser*).
* Implementation follows the paper: ``a_max_threshold =
  a_max_fraction * a_max`` (real ``float`` multiplication, see
  ``run`` below). This was a deliberate choice: the paper's
  mathematical formulation is the authoritative specification;
  the official-code ``//`` is an implementation accident (likely
  a transcription bug from ``/`` → ``//``) and produces no useful
  semantic difference for the heavy-tail regime the criterion
  targets, but yields a stricter, paper-faithful threshold here.

Output context slots
--------------------
Reads:
  * ``max_acc`` — ``dict[(int, int), float]``, per-expert
    down_proj max magnitude (collected by the shared calibration
    pass; see :class:`MADetectionPlugin` for the upstream).
  * ``L`` — ``set[int]``, MA-formation layers (from
    :class:`MADetectionPlugin`).
  * ``candidate_bag`` — shared :class:`CandidateBag` mutated in
    place by this and the three other candidate-generator plugins
    (``aimer``, ``sink_token``, ``magnitude_topk``).
  * ``config`` — for the ``a_max_fraction`` knob.

Writes:
  * ``p995`` — ``float``, the 99.5th-percentile threshold.
  * ``a_max`` — ``float``, the global max over the eligible set.
  * ``a_max_threshold`` — ``float`` = ``a_max_fraction * a_max``.
  * ``candidate_bag`` — mutated in place; each (l, e) passing the
    three-way AND is added with provenance tag ``"phase_c"``.

The ``p995`` / ``a_max`` / ``a_max_threshold`` triple is consumed by
the orchestrator-built ``blacklist_config`` block of
``stage1_blacklist.json``.

``provides`` is ``("downproj_max",)`` — the shared
:class:`CalibrationEngine` is asked to expose the per-expert
down_proj max magnitude accumulator so this plugin can read
``max_acc`` from ctx without running its own forward pass.

Artifact contribution: none (:meth:`contribute_artifact` returns
``{}``). The three-way AND statistics live inside the
orchestrator-built ``blacklist_config`` block; the candidate set
itself is recorded under ``aimer.candidates``, ``sink_token.candidates``,
and ``magnitude_topk.candidates`` (provenance lists), not under a
top-level ``three_way_and`` key — the 7-top-level-keys schema
invariant is preserved.

Naming-history note
-------------------
The legacy log strings ``Stage 1 Phase C₁ (three-way AND): ...``
are preserved as-is for Trackio dashboard compatibility. The
concern itself ("the load-bearing paper-Eq.-6 SE criterion") does
not need a Phase label in new code.
"""

from __future__ import annotations

import logging

import numpy as np

from ...pipeline.candidates import CandidateBag
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class ThreeWayAndPlugin:
    """Three-way AND criterion detector — paper Eq. 6 (load-bearing).

    See the module docstring for paper text + verified line refs +
    official-code citation + the D-SE-A, D-a-max-fraction, and
    D-amax-mult-real-vs-floor deviations.

    Mandatory paper criterion — :meth:`is_enabled` returns ``True``
    unconditionally; the three-way AND is the central SE-definition
    Eq. 6 of arXiv:2507.23279.

    Reads ``max_acc`` + ``L`` + ``candidate_bag`` + ``config`` from
    ctx; writes ``p995`` + ``a_max`` + ``a_max_threshold`` + mutates
    ``candidate_bag`` in place by adding tagged candidates with tag
    ``"phase_c"``.

    No artifact fragment is contributed (:meth:`contribute_artifact`
    returns ``{}``). The three-way AND statistics live inside the
    orchestrator-built ``config`` (``blacklist_config``) block of
    ``stage1_blacklist.json``, not under their own top-level key —
    the 7-top-level-keys schema invariant is preserved.
    """

    name: str = "three_way_and"
    paper: str = (
        "Su et al., 'Super Experts in MoE Models' (arXiv:2507.23279, 2025), "
        "Equation 6 — Super-Expert three-way AND criterion (`a_{l,e} > P99.5 "
        "AND a_{l,e} > (1/10)·a_max AND l ∈ L`). Official code: "
        "github.com/ZunhaiSu/Super-Experts-Profilling @ commit "
        "573aead3127ae593ba267758b832944f8fed1485 (2025-09-25), "
        "`eval_utils.py:619-651` (`_identify_super_experts`, "
        "`quantile=99.5`, `times=10`). Deviations: D-SE-A (A restricted "
        "to l ∈ L per Algorithm 1; resolves a prose-vs-pseudocode "
        "ambiguity); D-a-max-fraction (the 1/10 multiplier exposed as "
        "the `a_max_fraction` config knob, default 0.1 matches paper); "
        "D-amax-mult-real-vs-floor (real `0.1 * a_max` per Algorithm 1, "
        "not the official code's `np.max(...) // 10` floor-divide). "
        "See module docstring for full justifications."
    )
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

    def run(self, ctx: PipelineContext) -> None:
        """Compute (p995, a_max) statistics, write three slots, add candidates.

        Reads ``max_acc``, ``L``, ``candidate_bag``, ``config`` from
        ``ctx``. Writes ``p995``, ``a_max``, ``a_max_threshold`` to ctx
        (consumed downstream by ``AimerDetectorPlugin.run`` for its
        layer-max gating, and by the orchestrator for the
        ``blacklist_config`` block). Mutates ``candidate_bag`` in place
        by adding each (l, e) satisfying the three-way AND criterion
        with tag ``"phase_c"``.

        Empty-L semantics: if ``L`` is empty, ``_compute_se_thresholds``
        returns ``(0.0, 0.0)``, ``a_max_threshold = 0.0``, and
        ``_apply_paper_criterion`` early-returns ``{}`` (its first guard
        is ``if not L: return {}``), so no candidates are added. The
        three statistics are still written to ctx so the
        ``blacklist_config`` block reads valid floats.
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

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return ``{}`` — three-way AND statistics live in ``blacklist_config``.

        The orchestrator reads ``p995`` / ``a_max`` / ``a_max_threshold``
        from ctx and emits them under the ``config`` top-level key
        (which is orchestrator-owned, not plugin-owned). The plugin
        contributes no top-level fragment. Empty-dict return is
        Protocol-compliant.
        """
        return {}


# ---------------------------------------------------------------------------
# Three-way AND private helpers — sole caller: ThreeWayAndPlugin.run.
# (Concern previously known as "Phase C₁" of the pre-refactor Stage 1
# monolith; see naming-history note in the module docstring.)
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
