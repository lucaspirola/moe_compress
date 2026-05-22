"""Phase C₂ — AIMER (Activation-Independent Magnitude-Energy Ratio) detector.

Paper: arXiv:2603.18492 (AIMER). Migrated from the legacy Stage 1 module
in sub-task 6 of the Stage 1 → plugin-architecture refactor.

The plugin owns three responsibilities (sub-task 8 added the third):

1. **Pre-computation** — per-(layer, expert) AIMER scores from
   ``utils.aimer.aimer_score_tensor`` and per-layer bottom-pct selections
   from ``utils.aimer.aimer_bottom_pct_per_layer``. Written to the
   ``aimer_scores`` and ``bottom_pct_by_layer`` slots on the
   ``Stage1Context``.
2. **Candidate-pool contribution** — :meth:`run` also gates the
   bottom-pct selections by per-layer activation max (``max_acc`` ×
   ``a_max``) and adds each surviving (l, e) to the shared
   ``CandidateBag`` with tag ``"aimer"``. The gate inputs (``max_acc``,
   ``a_max``) travel via ctx — ``a_max`` is written by
   ``ThreeWayAndPlugin.run`` earlier in the Phase-C plugin sequence.
3. **Artifact contribution** — returns the three-key ``aimer`` block of
   ``stage1_blacklist.json`` via :meth:`contribute_artifact`. The
   ``candidates`` key is derived from the unified ``candidates`` dict
   materialised from the shared ``CandidateBag`` via
   ``bag.to_provenance_dict()``; the plugin only INVERTS that dict's
   tag list for the fragment.

The plugin's ``writes`` field is ``("aimer_scores", "bottom_pct_by_layer",
"candidate_bag")``. The ``candidate_bag`` slot appears in both ``reads``
and ``writes`` — it is read (the bag instance) and mutated in place via
``add(l, e, "aimer")``. See `tasks/refactor_stage1/subtask_8_plan.md`
§2.4 for the rationale on extending ``run`` to own the candidate-add
step.
"""

from __future__ import annotations

import logging

import torch

from .._framework.candidates import CandidateBag
from .._framework.plugin import StagePlugin  # noqa: F401  (Protocol import for type-checkers)
from .._framework.safe_json import safe_float
from ...utils.aimer import aimer_bottom_pct_per_layer, aimer_score_tensor
from ..context import Stage1Context

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
      with NaN/±Inf scrubbed to ``None`` via ``_framework.safe_json.safe_float``.
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
    paper: str = "AIMER: Activation-Independent Magnitude-Energy Ratio (arXiv:2603.18492)"
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
    accumulators: tuple[str, ...] = ()

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

    def run(self, ctx: Stage1Context) -> None:
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

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
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
# four expert-layout variants:
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
