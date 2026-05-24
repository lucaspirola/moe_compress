"""AIMER weight-only expert importance scorer (calibration-free, per-expert).

Paper
-----
Liu et al., "AIMER: Calibration-Free Task-Agnostic MoE Pruning",
arXiv:2603.18492 (2026). Acronym expansion (paper title + Section 3):
**AIMER = Absolute mean over root mean square IMportance for Expert
Ranking.**

Paper Section 3 defines the AIMER score in two equivalent forms.
Equation (4) writes it as a ratio of aggregates over the three
projection matrices of one expert (with ``W_gate, W_up ∈ ℝ^{m×d}``
and ``W_down ∈ ℝ^{d×m}``)::

    N = N_gate + N_up + N_down
    P = ‖W_gate‖₁ + ‖W_up‖₁ + ‖W_down‖₁
    Q = ‖W_gate‖²_F + ‖W_up‖²_F + ‖W_down‖²_F
    (paper Eq. 3 labels all three rows of the (N, P, Q) block)

    AIMER = (P / N) / sqrt(Q / N) = P / sqrt(N · Q)   (paper Eq. 4)

Equation (5) is the algebraically equivalent vector form, obtained
by flattening and concatenating the three projection matrices into a
single vector ``w ∈ ℝ^N``::

    AIMER(w) = ‖w‖₁ / (√N · ‖w‖₂)                    (paper Eq. 5)

i.e. the ratio of the absolute mean ``‖w‖₁ / N`` to the
root-mean-square ``‖w‖₂ / √N``. Paper Eq. (7) gives the bounds
``1/√N ≤ AIMER(w) ≤ 1``:

    upper bound 1.0  — all entries equal magnitude (most distributed)
    lower bound 1/√N — single non-zero entry (most concentrated)

Paper Algorithm 1 ("PyTorch-style expert ranking with AIMER", pages
4-5 of the arXiv PDF — the AIMER criterion section spans both, with
the algorithm box and Eq. (5)/(6)/(7) on page 5) implements the
score per expert by summing the abs /
square reductions over all three projections, then taking the ratio,
and ranks ``torch.sort(scores, descending=True)`` so the
HIGHEST-score (most-distributed) experts come first. **The paper
uses AIMER as a PRUNING criterion**: the sentence immediately
following Eq. (4) reads "and we prune experts with larger AIMER
scores" (most-distributed = lowest information density = safe to
remove).

Official implementation (golden reference)
------------------------------------------
``github.com/ZongfangLiu/AIMER`` pinned to commit
``fcf8e28f9253810bb117bc3a57c65e98780f4706`` (default branch HEAD,
``pushed_at`` = 2026-03-23). The reference scoring is in
``src/calib_free_prune.py::_aimer_scores_and_rank`` at that SHA
(verified by raw-blob fetch). The reference formula expands to::

    abs_sum = gate.abs().sum() + up.abs().sum() + down.abs().sum()
    numel   = gate.numel()    + up.numel()    + down.numel()
    l2_sq   = gate.square().sum() + up.square().sum() + down.square().sum()
    score   = (abs_sum / numel) / torch.sqrt(l2_sq / numel)

which is algebraically identical to Eq. (4)/(5) and to the project's
``utils.aimer.aimer_score_tensor`` (``l1 / (sqrt(n) * l2)``).

Two deliberate project deviations from the paper
------------------------------------------------
**(1) — Down_proj-only score** (this project) vs **concatenated
gate+up+down** (paper).

The shared scoring utility ``utils.aimer.aimer_score_tensor(w)``
applies Eq. (5) to whatever tensor ``w`` it receives — the FORMULA is
paper-exact. This plugin then calls it with **only** the expert's
``down_proj`` weight tensor (see :meth:`AimerDetectorPlugin.run` —
``aimer_scores[(ref.layer_idx, e)] = aimer_score_tensor(w_down)``),
**not** the gate+up+down combination prescribed by Algorithm 1 / Eq. (5).

Rationale (introduced in commit ``507a979`` "feat(stage1): AIMER
weight-only expert score utility"; preserved in S1-1 commit
``743073f`` when the package was ported): "down_proj weights have
concentrated energy (the structural signature of an SE that may
have been missed by the residual-stream-based detector)". Restricting
to ``down_proj`` aligns AIMER with the Super Experts paper's
formulation, where SE-defining magnitudes are measured on the
``down_proj`` output (arXiv:2507.23279 Algorithm 1 line 19 —
``a_{l,e} = max_{x∈D} |h_{l,e}(x) · W^{l,e}_{down_proj}|``). Cross-projection
mixing was judged less informative for SE-detection than for the
paper's task-agnostic pruning use case.

**(2) — AIMER repurposed as SE-CANDIDATE signal**, not as a pruning
criterion (D-aimer-cross-check).

The paper uses AIMER to PRUNE high-score (most-distributed) experts.
This project uses AIMER to PROTECT low-score (most-concentrated)
experts — bottom-``aimer_bottom_pct`` (= 1% by default) per layer
enter the Phase-C SE-candidate pool, gated by a per-layer activation-
max threshold. The two usages are operationally inverse but
mathematically dual: protecting the most-concentrated is the same
ranking as pruning the most-distributed under the AIMER ordering.

Final inclusion in ``stage1_blacklist.json`` requires ablation
evidence from :class:`AblationFilterPlugin` (paper has no equivalent
filter; both ``static-threshold detection`` and ``ablation filtering``
are project-original additions). The bottom-pct + activation-max
gate together keep the candidate-pool size small relative to a
layer-wide AIMER sweep.

Plugin responsibilities
-----------------------
1. **Pre-computation** — per-(layer, expert) AIMER scores via
   ``utils.aimer.aimer_score_tensor`` on ``down_proj`` weights, and
   per-layer bottom-pct selections via
   ``utils.aimer.aimer_bottom_pct_per_layer``. Written to the
   ``aimer_scores`` and ``bottom_pct_by_layer`` ctx slots.
2. **Candidate-pool contribution** — :meth:`run` gates the bottom-pct
   selections by ``per_expert_max[(l, e)] > aimer_layer_max_fraction
   · a_max`` (where ``a_max`` is written by
   :class:`ThreeWayAndPlugin`) and adds each surviving (l, e) to the
   shared ``CandidateBag`` with provenance tag ``"aimer"``.
3. **Artifact contribution** — returns the three-key ``aimer`` block
   of ``stage1_blacklist.json`` via :meth:`contribute_artifact`:
     * ``scores`` — every (l, e) score (NaN/Inf scrubbed to JSON null)
     * ``bottom_pct_per_layer`` — expert IDs per layer, lowest first
     * ``candidates`` — derived by inverting the shared CandidateBag
       on the ``"aimer"`` tag

Output context slots
--------------------
Reads:
  * ``moe_layers``, ``L``, ``config`` — model structure + MA-formation
    layer set + the ``aimer_bottom_pct`` + ``aimer_layer_max_fraction``
    config knobs.
  * ``max_acc`` — per-expert down_proj max magnitude (from the shared
    calibration pass).
  * ``a_max`` — global max written by :class:`ThreeWayAndPlugin`.
  * ``candidate_bag`` — shared :class:`CandidateBag`, mutated in place.

Writes:
  * ``aimer_scores``, ``bottom_pct_by_layer``, ``candidate_bag``.

``provides`` is empty — AIMER is weight-only; no activation
accumulator from the shared :class:`CalibrationEngine` is required.

Naming-history note
-------------------
The legacy log strings and the "Phase C₂" prefix in code comments
trace to the pre-refactor Stage 1 monolith; they are preserved
unchanged for Trackio-dashboard compatibility. The concern is
"AIMER candidate-source for SE detection", not "Phase C₂".
"""

from __future__ import annotations

import logging

import torch

from ...pipeline.candidates import CandidateBag
from ...pipeline.safe_json import safe_float
from ...utils.aimer import aimer_bottom_pct_per_layer, aimer_score_tensor
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class AimerDetectorPlugin:
    """AIMER weight-only detector (Phase C₂).

    Pre-computes per-(layer, expert) AIMER scores from each expert's
    ``down_proj`` weight tensor and the per-layer bottom-``aimer_bottom_pct``
    selection of experts (lowest score = most concentrated = most critical
    to keep unmerged). Both maps are written to the context.

    :meth:`run` also owns the candidate-add step (sub-task 8): it gates
    the per-layer bottom-pct selection by per-layer activation max and
    adds each surviving (l, e) to the shared ``CandidateBag`` with tag
    ``"aimer"``.

    Returns the ``aimer`` block of ``stage1_blacklist.json`` via
    :meth:`contribute_artifact`. The block has exactly three keys:

    - ``scores``: ``{f"L{li}E{e}": float | None}`` — every (l, e) score
      with NaN/±Inf scrubbed to ``None`` via ``pipeline.safe_json.safe_float``.
    - ``bottom_pct_per_layer``: ``{str(li): [int, ...]}`` — expert ids
      per layer, lowest score first.
    - ``candidates``: ``{str(li): sorted([int, ...])}`` — derived by
      inverting the unified Phase-C ``candidates`` dict (on the ctx,
      materialised from the shared ``CandidateBag``) on the ``"aimer"``
      tag.

    Scope note: as of sub-task 8 the plugin owns the AIMER candidate-add
    step end-to-end — :meth:`run` mutates the shared ``CandidateBag``
    rather than leaving the union to a legacy
    ``_collect_candidates``. The ``candidate_bag`` slot
    appears in both ``reads`` and ``writes``: it is read (the bag
    instance) and mutated in place. See sub-task 8 plan §2.4.
    """

    name: str = "aimer"
    paper: str = (
        "Liu et al., 'AIMER: Calibration-Free Task-Agnostic MoE Pruning' "
        "(arXiv:2603.18492, 2026). Acronym = Absolute mean over root mean "
        "square IMportance for Expert Ranking; paper Eq. (4) "
        "AIMER = P / sqrt(N·Q) over the three projections (Eq. 3 labels "
        "the P row of the (N, P, Q) definition block), equivalent to the "
        "vector form Eq. (5) "
        "AIMER(w) = ‖w‖₁ / (√N · ‖w‖₂) on the flattened concatenation. "
        "Official code: github.com/ZongfangLiu/AIMER @ commit "
        "fcf8e28f9253810bb117bc3a57c65e98780f4706 (pushed 2026-03-23) — "
        "reference scoring in src/calib_free_prune.py::_aimer_scores_and_rank. "
        "Two project deviations from the paper: (1) score computed on "
        "down_proj weights only (paper Algorithm 1 concatenates "
        "gate+up+down) — narrows AIMER to the SE structural signature; "
        "(2) D-aimer-cross-check — repurposed as SE-candidate signal "
        "(protect low-score / most-concentrated experts) rather than "
        "the paper's pruning criterion (prune high-score / most-"
        "distributed experts). See module docstring for full "
        "justifications + git-archaeology (commit 507a979 introduced "
        "the down_proj-only choice)."
    )
    config_key: str = "stage1_grape.super_expert_detection.aimer_enabled"
    reads: tuple[str, ...] = (
        "moe_layers",
        "L",
        "config",
        "max_acc",          # sub-task 8: for layer_expert_max + a_max gating.
        "a_max",            # sub-task 8: written by ThreeWayAndPlugin.run.
        "candidate_bag",    # sub-task 8: shared write surface.
    )
    writes: tuple[str, ...] = (
        "aimer_scores",
        "bottom_pct_by_layer",
        "candidate_bag",    # sub-task 8: mutated in place via .add(l, e, "aimer").
    )
    # AIMER is weight-only; no Phase-B activation accumulator is consumed.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Read ``config["stage1_grape"]["super_expert_detection"]["aimer_enabled"]``;
        default ``True``.

        ``False`` does **not** skip Phase C₂ entirely — the plugin still
        runs and writes empty ``aimer_scores`` / ``bottom_pct_by_layer``
        dicts (matching the legacy inline behaviour). ``is_enabled``
        reflects the orchestrator-visible flag for the orchestrator's
        gating.
        """
        s1 = config.get("stage1_grape", {})
        se = s1.get("super_expert_detection", {})
        return bool(se.get("aimer_enabled", True))

    def run(self, ctx: PipelineContext) -> None:
        """Execute Phase C₂ pre-computation + candidate-add.

        Reads ``moe_layers``, ``L``, ``config`` from ``ctx``; writes
        ``aimer_scores`` and ``bottom_pct_by_layer`` back. Then (sub-task
        8) gates the bottom-pct selections by per-layer activation max
        (reading ``max_acc`` + ``a_max`` + ``candidate_bag`` from ctx)
        and adds each surviving (l, e) to the shared ``CandidateBag``
        with tag ``"aimer"``.

        - ``aimer_scores``: ``{(layer_idx, expert_idx): float}``. Empty
          dict when ``aimer_enabled`` is ``False``.
        - ``bottom_pct_by_layer``: ``{layer_idx: [expert_idx, ...]}``
          (lowest score first). Empty dict when ``aimer_scores`` is empty
          (matches the legacy ``aimer_bottom_pct_per_layer(scores, pct)
          if scores else {}`` guard).

        Candidate-add semantics: the ``if aimer_enabled and
        bottom_pct_by_layer:`` guard short-circuits the candidate-add
        block when AIMER is disabled (no ``max_acc`` / ``a_max`` /
        ``candidate_bag`` read on that path) or when no bottom-pct
        candidates exist — byte-identical to the legacy
        ``_collect_candidates`` AIMER branch.
        """
        moe_layers = ctx.get("moe_layers")
        L: set[int] = ctx.get("L")
        config: dict = ctx.get("config")

        s1 = config["stage1_grape"]
        se_cfg = s1.get("super_expert_detection", {})
        aimer_bottom_pct = float(se_cfg.get("aimer_bottom_pct", 0.01))
        aimer_layer_max_fraction = float(se_cfg.get("aimer_layer_max_fraction", 0.1))
        aimer_enabled = bool(se_cfg.get("aimer_enabled", True))

        log.info(
            "Stage 1 Phase C₂ (AIMER): aimer_enabled=%s, bottom_pct=%.4f, "
            "MA-formation layers L=%s (informational; AIMER is weight-only).",
            aimer_enabled, aimer_bottom_pct, sorted(L) if L else [],
        )

        aimer_scores: dict[tuple[int, int], float] = {}
        if aimer_enabled:
            for ref in moe_layers:
                for e in range(ref.num_routed_experts):
                    w_down = _get_expert_down_proj_weight(ref, e)
                    aimer_scores[(ref.layer_idx, e)] = aimer_score_tensor(w_down)

        bottom_pct_by_layer: dict[int, list[int]] = (
            aimer_bottom_pct_per_layer(aimer_scores, pct=aimer_bottom_pct)
            if aimer_scores
            else {}
        )

        ctx.set("aimer_scores", aimer_scores)
        ctx.set("bottom_pct_by_layer", bottom_pct_by_layer)

        # Sub-task 8 addition: gate AIMER candidates by per-layer activation
        # max (the pre-sub-task-8 ``_collect_candidates`` in the legacy
        # Stage 1 module) and add each to the shared ``CandidateBag``
        # with tag "aimer". The gating uses ``max_acc.per_expert_max`` (Phase B
        # output) and ``a_max`` (written by ThreeWayAndPlugin.run earlier in
        # the Phase-C plugin sequence). Logic moved verbatim.
        if aimer_enabled and bottom_pct_by_layer:
            max_acc = ctx.get("max_acc")
            a_max: float = ctx.get("a_max")
            candidate_bag: CandidateBag = ctx.get("candidate_bag")

            per_expert_max = max_acc.per_expert_max
            layer_expert_max: dict[int, float] = {}
            for (li, _e), v in per_expert_max.items():
                layer_expert_max[li] = max(layer_expert_max.get(li, 0.0), v)
            for li, exps in bottom_pct_by_layer.items():
                if layer_expert_max.get(li, 0.0) <= aimer_layer_max_fraction * a_max:
                    continue
                for e in exps:
                    candidate_bag.add(int(li), int(e), "aimer")

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the three-key ``aimer`` block of ``stage1_blacklist.json``.

        Identical schema to the legacy inline construction (pre-sub-task-6):

        Returns
        -------
        dict
            Exactly three top-level keys:
              - ``scores`` : ``{f"L{li}E{e}": float | None}`` — every
                ``(layer_idx, expert_idx)`` from ``aimer_scores`` with
                non-finite floats scrubbed to ``None``.
              - ``bottom_pct_per_layer`` : ``{str(li): [expert_idx, ...]}``
                — lowest-score first per layer.
              - ``candidates`` : ``{str(li): sorted([expert_idx, ...])}``
                — derived from the unified ``candidates`` dict on the ctx
                (materialised from the shared ``CandidateBag`` via
                ``to_provenance_dict()`` BEFORE the delegation block
                calls this method).
        """
        aimer_scores: dict[tuple[int, int], float] = ctx.get("aimer_scores")
        bottom_pct_by_layer: dict[int, list[int]] = ctx.get("bottom_pct_by_layer")
        candidates: dict[tuple[int, int], list[str]] = ctx.get("candidates")

        # Invert ``candidates`` on the "aimer" tag — byte-identical to the
        # legacy inline ``_candidates_by_provenance("aimer")`` (pre-sub-task-6).
        aimer_candidates_by_layer: dict[int, list[int]] = {}
        for (li, e), tags in candidates.items():
            if "aimer" in tags:
                aimer_candidates_by_layer.setdefault(int(li), []).append(int(e))

        return {
            "scores": {
                f"L{li}E{e}": safe_float(v)
                for (li, e), v in aimer_scores.items()
            },
            "bottom_pct_per_layer": {
                str(li): list(exps) for li, exps in bottom_pct_by_layer.items()
            },
            "candidates": {
                str(li): sorted(es) for li, es in aimer_candidates_by_layer.items()
            },
        }


# ---------------------------------------------------------------------------
# Per-expert weight accessor (moved verbatim from the legacy Stage 1 module
# in sub-task 6). Sole caller is :class:`AimerDetectorPlugin.run`. Handles
# five expert-layout variants:
#  1. Per-expert ModuleList (legacy) — ``experts[e].down_proj.weight``.
#  2. Fused stacked nn.Parameter — ``experts.down_proj[e]`` of shape
#     ``(num_routed_experts, d_hid, d_int)``.
#  3. Fused stacked nn.Module — ``experts.down_proj.weight[e]``.
#  4. FactoredExperts U/V decomposition — ``U[e] @ V[e]`` reconstruction.
#  5. Last-resort accessor — ``experts.down_proj_weight(e)`` callable.
# ---------------------------------------------------------------------------


def _get_expert_down_proj_weight(ref, expert_idx: int) -> torch.Tensor:
    """Return the down_proj weight tensor for a single expert in this MoE layer.

    Supports both fused (FactoredExperts / Qwen3_5MoeExperts-style stacked
    parameter) and per-expert module layouts. Used by the AIMER weight-only
    score in Phase C₂ where we need per-expert ``W_down``.
    """
    experts = ref.experts_module
    # Per-expert layout (legacy ModuleList of nn.Modules with a .down_proj submodule).
    if hasattr(experts, "__len__") and not isinstance(experts, torch.nn.Linear):
        try:
            return experts[expert_idx].down_proj.weight
        except (TypeError, AttributeError, IndexError):
            pass
    # Fused layout: experts.down_proj is a stacked nn.Parameter
    # of shape (num_experts, ...).
    fused = getattr(experts, "down_proj", None)
    if fused is not None:
        if isinstance(fused, (torch.nn.Parameter, torch.Tensor)):
            if fused.shape[0] == ref.num_routed_experts:
                return fused[expert_idx]
        elif hasattr(fused, "weight"):
            w = fused.weight
            if w.shape[0] == ref.num_routed_experts:
                return w[expert_idx]
    # FactoredExperts layout: down_proj is decomposed into U @ V.
    # Reconstruct the effective down_proj weight as U @ V (shape (d_hid, d_int)).
    u = getattr(experts, "down_proj_U", None)
    v = getattr(experts, "down_proj_V", None)
    if (u is not None and v is not None
            and isinstance(u, (torch.nn.Parameter, torch.Tensor))
            and isinstance(v, (torch.nn.Parameter, torch.Tensor))):
        return u[expert_idx] @ v[expert_idx]
    # Last-resort: ref.down_proj_weight(expert_idx) if such an accessor exists.
    accessor = getattr(experts, "down_proj_weight", None)
    if callable(accessor):
        return accessor(expert_idx)
    raise AttributeError(
        f"Could not locate down_proj weight for expert {expert_idx} in layer {ref.layer_idx}; "
        f"experts_module type: {type(experts).__name__}"
    )
