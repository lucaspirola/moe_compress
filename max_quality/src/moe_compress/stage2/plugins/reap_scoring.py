"""REAP per-layer expert-saliency scoring (Eq. 9).

Paper
-----
Thangarasa et al., "REAP the Experts: Why Pruning Prevails for One-Shot
MoE Compression" — arXiv:2510.13999 (ICLR 2026).
audit/spec_compliance/01_papers/2510.13999/source.md.

Equation 9 (per-expert saliency, the centroid-candidacy signal):

    S_j = (1/|X_j|) · Σ_{x ∈ X_j} g_j(x) · ‖f_j(x)‖₂

where ``X_j = {x | j ∈ TopK(σ(x))}`` is the set of calibration tokens on
which expert ``j`` is dispatched, ``g_j(x)`` is the post-softmax routing
weight, and ``f_j(x)`` is the expert output.

Official code
-------------
``CerebrasResearch/reap`` @ commit
``1970473c51ca3caeb98c10392f15b3a08a672974`` (2026-04-17) —
github.com/CerebrasResearch/reap. The reference implementation is the
basis for the REAP scoring pass.

Deviation: D-reap-routing-weight
--------------------------------
Paper Eq. 9 is silent on whether ``g_j(x)`` is the un-renormalized
masked softmax or the dispatched (top-k renormalized) weight. The
plugin uses the **dispatched** weight as the model's forward pass
applies it — for Qwen3-MoE this is
``softmax(router_logits)[j] / Σ_{k∈top-k} softmax(router_logits)[k]``
(top-k softmax outputs renormalized to sum=1 over the top-k set).

Rationale: the model's actual forward output uses the renormalized
top-k weight, so REAP's "expert importance" ``S_j`` is most faithful to
the model's behavior when computed against the same weight the experts
actually receive. Both readings (renormalized vs un-renormalized) are
defensible from Eq. 9 alone. The un-renormalized reading would yield
the same expert *ranking* only if the per-token sum is constant, which
it is not (varies per token), so the choice is empirically
distinguishable — just not in a paper-prescribed direction.
Aligns with the upstream CLI default
(``renormalize_router_weights=True`` at
``CerebrasResearch/reap`` ``src/reap/args.py:142``).

Deviation: D-reap-min-active-tokens
-----------------------------------
REAM/REAP do not describe a minimum-active-tokens filter for centroid
candidacy. The plugin's downstream ``select_centroids_by_reap``
implements ``reap_min_active_tokens`` (configurable; default ``0`` in
code, set to ``32`` in the production config) that excludes experts
with fewer than 32 active calibration tokens from centroid candidacy.
Filtered experts become non-centroids and are merged.

Rationale: low-frequency experts have noisy gate/expert profiles
(averaged over <32 tokens), so promoting them to centroids would
propagate that noise into the merged weights. Filtering them to
non-centroid status routes them through the Hungarian alignment
(which projects them onto a higher-frequency centroid's neuron space)
instead. Compression target may shrink slightly when many low-frequency
experts are filtered; a WARNING is logged but **no compensating
budget-bump fires** — surfaces under-target compression without
silently absorbing it.

Calibration deviations (SHARED — also applies to Stage 2.5 / 5)
---------------------------------------------------------------
- **D11 — calibration data source**: REAP (2510.13999 §5) uses
  c4 + evol-codealpaca. The project uses multi-domain
  Nemotron-Cascade-2-SFT-Data with weighted subsets
  (chat 0.56, math 0.21, science 0.11, ...). Task-aware calibration
  better matches target deployment distribution.
- **D-cal-size — calibration sequence count**: REAM 2604.04356 §5 uses
  3072 sequences × 512 tokens (1.57 M tokens); REAP 2510.13999 uses
  1024 sequences × 2048 tokens (2.1 M tokens). The project uses
  4000 sequences × 2048 tokens (8.19 M tokens) — 5.2× / 3.9× more
  tokens; the longer 2048-token sequences match the deployment context
  length.

Routing-weight notation: REAP / REAM convention
-----------------------------------------------
- REAP (2510.13999) uses ``g_j(x)`` for the post-softmax routing
  weight, **masked** to zero for non-top-k experts.
- REAM (2604.04356) uses ``σ(x)_j`` for the **full unmasked softmax**
  (always strictly positive for every expert on every token).

This plugin owns the REAP-side ``g_j(x)`` view. The REAM cost plugin
(:mod:`stage2.plugins.ream_cost`) owns the ``σ(x)_j`` view for its
δ̃_expert numerator. Both views derive from the same per-token
router-logits forward pass.

Output context contract
-----------------------
- ``reads``: ``layer_ref``.
- ``writes``: ``reap_acc`` (the per-layer ``ReapAccumulator``),
  ``scores`` (per-expert saliency ``np.ndarray``), ``freq`` (per-expert
  token count ``dict[int, int]``).
- ``provides``: ``("reap_acc",)`` — declarative metadata for the
  orchestrator's calibration-pass wiring.

Two phase-hooks: ``on_layer_setup`` (instantiate the per-layer
``ReapAccumulator`` and stash it on the context); ``on_score``
(finalize the accumulator and publish ``scores`` + ``freq``).

``contribute_artifact`` returns ``{}`` — REAP scores feed in-memory
into the downstream cost plugins; nothing is written to disk by this
plugin.

Naming-history note
-------------------
The legacy stage-2 monolith called this "Phase: Step 1 REAP Scoring"
(per §5 Step 1). Plugin architecture has no phase taxonomy. New prose
drops the labels; the existing log lines and Trackio keys retain the
legacy names for dashboard back-compat.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from ...pipeline.context import PipelineContext
from ...utils.activation_hooks import ReapAccumulator

log = logging.getLogger(__name__)


class ReapScoringPlugin:
    """REAP per-expert saliency scoring; publishes ``scores`` and ``freq``.

    Implements REAP Eq. 9 (arXiv:2510.13999) — runs at on_score time
    between profile and compute_assignment. See module docstring for
    the official-code SHA, the D-reap-routing-weight deviation
    (renormalized top-k weight chosen as the runtime-faithful reading),
    the D-reap-min-active-tokens filter, and the shared calibration
    deviations D11 / D-cal-size.
    """

    name = "reap_scoring"
    paper = (
        "REAP Eq. 9: S_j = (1/|X_j|)·Σ g_j(x)·‖f_j(x)‖₂ — "
        "arXiv:2510.13999 (Thangarasa et al., ICLR 2026). "
        "Official code: CerebrasResearch/reap @ "
        "1970473c51ca3caeb98c10392f15b3a08a672974. "
        "Deviations: D-reap-routing-weight, D-reap-min-active-tokens; "
        "calibration: D11 + D-cal-size. See module docstring."
    )
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
