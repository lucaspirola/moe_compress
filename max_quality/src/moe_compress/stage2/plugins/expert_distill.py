"""Per-merge-group expert distillation against routing-gated original outputs.

Paper
-----
Inspiration:

- MoE-Pruner (arXiv:2410.12013) — **expert-wise** distillation, NOT
  block-level. Its Eq. 10 reads
  ``L_KD = L_CE + λ · Σ_{j,i} MSE(E_i^{j,teacher}, E_i^{j,student})``,
  i.e. a sum of per-expert MSEs between the (unpruned) teacher expert
  and the same-index (pruned) student expert across layers ``j`` and
  experts ``i`` (plus an LM cross-entropy term). The teacher of expert
  ``i`` is the pretrained unpruned expert ``i``.
- SlimMoE (arXiv:2506.18349) — top-8-logits KL on the **full model
  next-token distribution** (Eq. 1 / Eq. 3), NOT on MoE-block outputs.
  KL is computed between ``p_{teacher, top-8}(X)`` and ``p_W(X)``,
  with router updates concurrent across the multi-stage schedule.

Baseline REAM (arXiv:2604.04356) does NOT have a post-merge distill
step; the merge formula is a one-shot weighted average and no further
refinement.

Official / third-party code
---------------------------
The MoE-Pruner paper cites ``github.com/yanyue-xie/moe-pruner`` as the
official repo; that URL returns HTTP 404 (verified 2026-05-28) and no
authors' code has been located. The closest **third-party**
re-implementation is ``github.com/tanganke/fusion_bench``
(MIT-licensed, by Anke Tang, no formal affiliation with the paper's
authors) under
``fusion_bench/method/moe_pruner/`` — but it implements only the
Wanda-style pruning pass; a repo-wide ``grep -r distill
fusion_bench/method/moe_pruner/`` returns ZERO matches, so the paper's
Eq. 10 expert-wise distillation has no public reference implementation
to align against. Treat the paper text as authoritative; treat
``tanganke/fusion_bench`` as a *paper-grounded cross-check*, not a
paper-authoritative source.

Even ignoring that gap, no project-aligned per-merge-group
implementation exists upstream: MoE-Pruner's expert-wise loss pairs
teacher expert ``i`` with student expert ``i`` across the
unpruned/pruned bank (one-to-one same-index pairing), whereas this
plugin pairs the merged centroid with the *additive contribution of
its pre-merge group members*. The ``_distill_merged_group`` loop is
project-original.

Deviation: D-expert-distill-mse
-------------------------------
Stage 2 v2 adds ``expert_distill_steps`` (default ``0``) of AdamW MSE
distillation per non-singleton group. Target = routing-gated additive
contribution of pre-merge group members:

    target = Σ_{e ∈ g, e ∈ TopK(σ_orig(x))} g_e^orig(x) · E_e^orig(x)

(on tokens from ``X_g``; pre-merge router used). Student =
``g_g^merged(x) · E_g^merged(x)`` with the post-resize router row
frozen. Trainable: only the merged centroid's gate / up / down.
Plateau early-break, fp32 optimizer with bf16 forward, bank dtype
preserved on writeback.

Two project-original differences vs. SlimMoE / MoE-Pruner:

  (a) **Additive-target form against pre-merge MEMBERS within the
      merged group.** MoE-Pruner is expert-wise but its target is the
      *teacher's same-index expert* (one-to-one expert correspondence
      across teacher/student banks); SlimMoE distills at the full
      next-token distribution level. We have neither structure:
      experts are *merged into a centroid*, so there is no
      same-index teacher expert, and we deliberately avoid the
      block / model-level form to keep gradient attribution local to
      the centroid. Instead, the centroid is trained to reproduce the
      additive (freq- or routing-weighted) sum of the pre-merge
      members of its own group. This per-group additive target is
      what is novel — it is neither MoE-Pruner's per-expert pairing
      nor SlimMoE's full-model logit KL.
  (b) Expert-only training is strictly separated from router-only
      training (Stage 2.5) for resume-isolation and stage-boundary
      clarity. SlimMoE's distillation phases update both router and
      experts concurrently; MoE-Pruner's fine-tunes the full pruned
      model end-to-end.

The pre-merge router row is carried over verbatim to the post-resize
router (centroid expert's original row), so ``g_g^merged(x)`` is the
original centroid's routing weight evaluated under the new (smaller)
softmax denominator — Stage 2.5 retrains it. **Stage 2.5 consequently
sees a model whose merged centroids are already distilled — its job
becomes purely router calibration on top of pre-distilled experts,
not expert recovery (see § 5.5 of the Stage 2.5 plugin's docstring).**

Deviation: D-expert-distill-ce-term (Lift 1, paper Eq. 10)
----------------------------------------------------------
MoE-Pruner Eq. 10 (paper line 394) is

    L_KD = L_CE + λ · L_expert = L_CE + λ · Σ_{j,i} MSE(E_i^{j,t}, E_i^{j,s})

with ``L_CE`` the standard cross-entropy loss (paper line 410:
"LCE is the cross entropy loss") and ``λ`` "a weighting coefficient and
initialized based on the strength of cross entropy loss and expert-wise
knowledge distillation loss" (paper line 414). The paper's ``L_CE`` is
the LM next-token cross-entropy from the end-to-end fine-tune; this
plugin runs PER-LAYER PER-GROUP with no LM-head / labels in scope, so
the faithful adaptation is a **feature-level KL** between the
``softmax``-normalized teacher signal (the pre-merge group-member
additive forward — the SAME tensor the MSE term targets, already in
scope via ``pre_merge_weights``) and the ``log_softmax``-normalized
student (the merged centroid forward). Reduction is ``batchmean`` over
tokens.

Config knobs (Pattern C — consumed verbatim, no implicit coupling):
- ``expert_distill_use_ce_term`` (default True post-lift, False on
  v1-back-compat / A0..A11 ablation): when True the per-step loss is
  ``feature_KL + ce_lambda · MSE`` where ``feature_KL`` is the
  in-plugin feature-level KL (NOT paper-faithful Eq. 10 ``L_CE`` — see
  RESOLVED block below). When False the per-step loss is pure MSE,
  byte-identical to pre-lift behavior at the optimizer level.
- ``expert_distill_ce_lambda`` (default 1.0): parity weight between
  the in-plugin feature-KL term and the MSE term. Not paper Eq. 10's
  λ (which weights the deferred vocab-level ``L_CE`` against the same
  MSE — different magnitude regime). Tune via config; 1.0 is the safe
  parity default for the current adaptation.

RESOLVED: feature-level KL stays as a per-merge-group local term;
paper-faithful L_CE is DEFERRED to a separate downstream fine-tune phase
---------------------------------------------------------------------
**Decision (2026-05-28): option (2) — paper-faithful ``L_CE`` will
live in a SEPARATE downstream model-level fine-tune phase, NOT in this
plugin.** The in-plugin term implemented above (``_feature_kl_ce``,
Lift 1) stays as a local per-merge-group recovery signal — it is an
engineering adaptation, not a paper-faithful ``L_CE``, and it is
explicitly documented as such here.

Why the obvious paper-literal path was rejected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The intuitive "bubble teacher LM-head logits into ctx + project
student through the downstream stack" path (call it option (a)) is
mathematically clean but **catastrophically expensive** at this
plugin's per-step granularity. Two independent agents confirmed the
blocker:

  - Plan branch ``plan/moe-pruner-ce-term`` (commit ``f15ff1d``,
    "plan(moe-pruner): paper Eq. 10 CE term — option (a) BLOCKED
    writeup") — structured §2 cost analysis: each gradient step would
    need to forward the trained merged centroid's ``(T, hidden)``
    output through ~``94 - N`` remaining MoE decoder layers (worst
    case ~47 at the mid-stack), RMSNorm, and the LM head to reach
    vocab space. At ~75 ms/layer for 8192 tokens BF16 on a single
    H100, that is ~3.5 s/step vs. the current ~7 ms/step — a ~500×
    per-step regression. Stage-2-total wall projection moves from
    ~3h to ~60d on a single H100, fully outside the project's
    wall-clock budget. Also requires the full pruned student model
    resident on the same GPU (~30 GB BF16) and activation
    checkpointing plumbing that does not exist in Stage 2 today.
  - A subsequent direct-implementer pass independently reached the
    same blocker on first contact with ``_distill_merged_group`` and
    halted with a writeup rather than push a multi-day-Stage-2
    regression (local-only branch, not preserved); the canonical
    record of the cost analysis and 5-option decision space lives at
    ``plan/moe-pruner-ce-term`` (commit ``f15ff1d``).

Per CLAUDE.md "RAISE, don't substitute"
(``feedback_raise_dont_substitute.md``), the user was given the
decision boundary and picked option (2): defer ``L_CE`` to a downstream
fine-tune phase that has the LM head + labels already in scope.

Status of the in-plugin term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``_feature_kl_ce`` helper (Lift 1) is a **feature-level KL** on the
per-layer per-group ``(T, hidden)`` tensors, NOT paper-faithful
``L_CE``. Two concrete semantic gaps remain and are accepted as the
cost of the local-only design:

  1. The hidden-dim softmax has no meaningful event space at the
     feature level. The paper's ``L_CE`` is a categorical cross-entropy
     over vocabulary tokens (one event per vocab slot); a softmax over
     a hidden-state's feature axis manufactures a distribution where
     the "events" are arbitrary coordinate indices.
  2. The hidden-axis softmax is **invariant to constant shifts along
     that axis**: any two hidden vectors ``h`` and ``h + c·1`` produce
     identical softmaxes, so the KL is blind to constant-shift errors
     in the merged centroid's output even though those shifts DO
     change downstream LM logits. (The MSE term still penalizes the
     shift, so the composite is non-degenerate in practice — but the
     CE contribution itself is shift-blind.)

The current implementation is still useful: it produces a
correctly-signed gradient component that pushes the merged centroid
toward the pre-merge group-member forward in distribution-shape (not
just magnitude). The ``ce_lambda=1.0`` default was chosen for
MSE-magnitude parity under this feature-level adaptation; it is a
parity weighting for the **in-plugin** term, NOT for paper Eq. 10's
``L_CE``.

Future fine-tune phase (deferred, NOT this plugin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When/if the downstream paper-faithful ``L_CE`` fine-tune phase lands,
it should consume the existing Path-B teacher cache
(``_stage5_teacher_logits.pt``, produced by
``max_quality/hf_jobs/precompute_teacher_logits.py``, schema_version=1,
manifest-last) directly — that artifact already covers the vocab-axis
teacher half of Eq. 10. No NEW calibration capture is required beyond
what Path-B already provides; the new infrastructure that DOES need to
be built is the end-to-end fine-tune harness itself (LM head + labels +
optimizer over the full pruned model), which is out of scope for the
Stage 2 per-merge-group plugin family.

Option (b) (LN-aware / shift-broken KL on the existing hidden tensors —
e.g. row-LayerNorm-aware KL that centers + scales hidden vectors so
the constant-shift invariance is broken in a structured way before the
softmax) stays available as a future **LOCAL** improvement to this
plugin's in-plugin term. It is distinct from the deferred paper-faithful
fine-tune phase: option (b) refines the local KL inside this plugin;
the deferred phase replaces the local KL with a true vocab CE in a
different stage. They are not alternatives — both can land
independently.

Deviation: D-expert-distill-paper-lift (Lift 2, paper Eqs. 1-3)
---------------------------------------------------------------
v2 target — adds the **TopK gate + per-token routing weight** that
v1 (D-expert-distill-mse-v1) drops. The paper's Eq. 1 (line 133) is

    y = Σ_i Gate(x)_i · E_i(x)

with ``Gate(x) = Softmax(TopK(x · W_g))`` (Eq. 2, line 142), i.e. the
post-top-k *renormalized* softmax over routed experts (TopK sets
non-top-k logits to ``−∞`` per line 144-145, so they vanish after
softmax). Eq. 3 (line 152) specializes this to top-2 for Mixtral.

Applied to the per-merge-group additive target, v2 becomes

    target_per_token = Σ_{e ∈ members ∩ TopK(σ_orig(x))}
                            g_e^orig(x) · E_e^orig(x)

where:
- ``σ_orig`` is the **pre-merge** router (recomputed from
  ``layer_ref.router.weight``; the pre-merge router row is the row
  the post-resize router carries verbatim for the centroid, so
  ``σ_orig`` evaluated on the unpruned router gives the original
  routing distribution — exactly what the paper's ``Gate(x)`` would
  produce);
- ``TopK`` is the layer's dispatch-time top-k (``layer_ref.top_k``);
- ``g_e^orig(x)`` is the renormalized post-top-k softmax weight per
  Eq. 2.

Tokens for which the group's members contribute ZERO routing mass
(no member is in the per-token top-k) carry a zero target — those
tokens still appear in the MSE / CE mean but with target = 0, which
is the paper-correct semantic: the merge cannot degrade what the
pre-merge router would not route through this group anyway.

Config knob (Pattern C — consumed verbatim):
- ``expert_distill_target_version``: ``"v1"`` | ``"v2"``, default
  ``"v2"`` post-lift. ``"v1"`` preserves the legacy freq-weighted
  target (D-expert-distill-mse-v1) for back-compat and A0..A11
  ablation parity; ``"v2"`` enables this paper-faithful target.

The v2 target is reconstructed from ``layer_ref.router`` (Wg, bias,
e_score_correction_bias) and the pre-merge expert weights in
``pre_merge_weights`` — no additional plumbing beyond the existing
v1 inputs is needed. The v1-waste of snapshotting all experts
(documented in ``_snapshot_pre_merge_layer_experts``) is unchanged.

Performance disclosure (v2 cold-cache CI cost)
----------------------------------------------
v2 adds a per-group ``_router_routing_weights`` recompute (full-softmax
``σ_orig(x)`` over the unpruned router, then a TopK mask and
renormalization) inside ``_distill_merged_group``. Empirical wall-clock
characteristics observed on the assignment_v2 test suite
(``tests/test_stage2_assignment_v2.py``):

- Main branch baseline: 52 passed, 6 skipped in ~3.7s
- This branch, cold cache: ~120s total with occasional
  ``--timeout=60`` failures
- This branch, warm cache: 1.5-13s (no regression)

The cold slowdown is NOT pre-existing — it is the cost of the per-group
router-routing-weight recompute, paid once per group on the first
distillation pass through a layer. Warm runs amortize via Python's
import cache + torch's kernel cache and show no measurable regression.

Post-fix (E-2): the per-group recompute has been lifted to a per-layer
cache in ``ExpertDistillPlugin.merge`` — see ``_build_v2_router_cache``
(this module). ``ExpertDistillPlugin.merge`` builds the
``(x_all, σ_orig, gate)`` triple ONCE at layer entry and threads it
into every ``_distill_merged_group`` call via the ``v2_router_cache``
kwarg, dropping the per-group cost from O(num_groups) to 1. The
pre-fix cold-cache wall-clock numbers above (~120s with intermittent
``--timeout=60`` failures) stand as the historical anchor; the cached
path collapses that overhead to one softmax+topk+linear per layer.
Direct helper callers (tests, ad-hoc usage) that omit
``v2_router_cache`` continue to hit the legacy in-helper recompute for
byte-identity with the pre-fix call signature.

Deviation: D-expert-distill-mse-v1
----------------------------------
The contract above is the *target*. The v1 implementation in
``_distill_merged_group`` simplifies for engineering tractability in
two ways — both coupled departures from the spec target, not minor:

  (i) Target uses **freq-weighted-only** mixing
      ``Σ (freq_e / Σ freq) · E_e^orig(x)`` — drops both the TopK
      gate (``e ∈ TopK(σ_orig(x))``) AND the per-token routing
      weight ``g_e^orig(x)``.
  (ii) Input tokens are the **reservoir-sampled layer-input** captured
       during profile (cap at ``expert_distill_token_cap = 8192``,
       seeded per-layer for reproducibility), not the routing-restricted
       ``X_g`` set.

Rationale: the full routing-gated form requires storing
``g_e^orig(x)`` per ``(expert, token)`` pair (additional memory) and
reconstructing ``X_g`` from ``ReamCostAccumulator.gate_logit_profiles``
keys (additional plumbing). v1 produces a correctly-signed
merge-error gradient on a uniform-token sample — the merged centroid
is still pulled toward a freq-weighted average of original-expert
outputs. **The gap between v1 and the spec target can be empirically
substantial** (the dropped TopK gate means group members that are not
even activated on a token still contribute to its target; the dropped
per-token weight flattens token-level importance) — Phase 3 v2 will
lift both simplifications, and the STRATEGY_NEXT § 8 ablation matrix
row A8 measures v1, A8' (planned) measures the spec form. Track A8
vs A8' separately when reporting.

Wiring
------
``ExpertDistillPlugin`` is LIVE as of S2-11: it owns the per-merge-group
expert distillation on the decomposed phase walk. Its
``pre_merge_snapshot`` hook snapshots the pre-merge expert weights and
its ``merge`` hook runs the ``_distill_merged_group`` loop (between
``_merge_experts_inplace`` and ``bank.select``). The orchestrator
registers it AFTER ``LegacyAdapter`` so its ``merge`` phase runs after
the adapter's ``_merge_experts_inplace``. ``LegacyAdapter.pre_merge_snapshot``
is now a no-op and its ``merge`` no longer distills (it only sets a
``distill_state=None`` default that this plugin overwrites).
``registry.enabled`` drops this plugin when ``expert_distill_steps`` is
0.

Circular-import note: this module imports only
``moe_compress.utils.model_io``, ``pipeline.base``, ``pipeline.context``
and ``pipeline.plugins.output_space_cost`` (for ``_swiglu_forward``) —
none of which import ``stage2_reap_ream`` or ``expert_distill``. No
cycle at module load, and every import below is a plain module-top
import (no function-scope late imports).

Back-compat / naming-history note
---------------------------------
"M8" / "step 7b" / "Phase 3 of the Stage 2 v2 plan" are STRATEGY_NEXT
labels. The current plugin architecture has no module-letter taxonomy;
new prose drops the labels. Existing log lines / Trackio keys preserve
those names for dashboard back-compat — this is the single canonical
disclaimer (referenced once, not repeated downstream).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks
from ...pipeline.context import PipelineContext
from .output_space_cost import _router_routing_weights, _swiglu_forward

log = logging.getLogger(__name__)


# ===========================================================================
# Phase 3 — per-merge-group expert distillation (spec § 5 step 7b / M8)
# ===========================================================================


def _feature_kl_ce(
    student: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-token soft-KL between ``softmax(target)`` and
    ``log_softmax(student)`` over the hidden dim.

    Per-layer per-group LOCAL adaptation, NOT paper-faithful ``L_CE``
    (see module docstring "RESOLVED" block). The paper computes
    ``L_CE`` as the LM next-token cross-entropy on the end-to-end
    fine-tune (paper line 410: "LCE is the cross entropy loss"); the
    per-layer per-group surface has no LM head / labels in scope, so
    we use a feature-level KL between the teacher (pre-merge
    group-member forward) and the student (merged centroid forward)
    on the same per-token ``(T, hidden)`` tensors instead. This is a
    documented engineering adaptation; paper-faithful ``L_CE`` is
    deferred to a separate downstream fine-tune phase that consumes
    the Path-B teacher cache directly (see module docstring).

    Reduction: ``batchmean`` over tokens, matching torch's
    ``F.kl_div`` convention. Returns a 0-D tensor.

    Both tensors are ``(T, hidden)`` float32. The function is
    invariant to constant shifts of either argument's last dim (KL on
    softmaxed distributions). Numerical stability comes from
    ``log_softmax`` on the student side; the teacher softmax is
    ``no_grad`` because the target is a constant w.r.t. the optimizer.
    """
    with torch.no_grad():
        p_teacher = F.softmax(target, dim=-1)
    log_p_student = F.log_softmax(student, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean")


def _build_v2_router_cache(
    *,
    layer_ref: MoELayerRef,
    layer_inputs: torch.Tensor,
    token_cap: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Per-layer cache of ``(x_all, σ_orig(x), gate)`` for v2 distillation.

    Closes E-2: the v2 ``_distill_merged_group`` path used to recompute
    ``σ_orig(x)`` + TopK-mask + renormalize ONCE PER CENTROID GROUP, even
    though every tensor in the triple is centroid-independent (the
    pre-merge router and the layer's reservoir-sub-sampled tokens are
    both frozen for the whole ``merge`` invocation). This helper hoists
    the computation to layer entry — ``ExpertDistillPlugin.merge`` builds
    the cache once and threads it into every per-group ``_distill_merged_group``
    call via the ``v2_router_cache`` kwarg, dropping the per-group
    re-execution from ``O(num_groups)`` to ``1``.

    Determinism contract: the cached ``x_all`` is byte-identical to the
    tensor ``_distill_merged_group`` would have built internally — same
    ``randperm(layer_idx)`` seed, same ``token_cap`` truncation, same
    ``.to(device, dtype=fp32)`` cast. The cached ``sigma`` is the full
    softmax over the pre-merge router (``_router_routing_weights``);
    ``gate`` is the renormalized TopK softmax per paper Eq. 2. All three
    tensors are detached / no-grad — autograd never touches them.

    Cache payload (all fp32, on ``device``):

    - ``x_all``: ``(T, hidden)`` — the deterministically sub-sampled
      tokens shared across every centroid in this layer.
    - ``sigma``: ``(T, n_experts)`` — full pre-merge router softmax.
    - ``gate``:  ``(T, n_experts)`` — renormalized TopK softmax mask.

    Memory footprint at default ``token_cap=8192`` / ``hidden=2048`` /
    ``n_experts=256``: ``x_all``≈64 MB, ``sigma``≈8 MB, ``gate``≈8 MB —
    ~80 MB live during the layer's distillation pass, freed at next
    layer's ``merge`` call (stack-local; no ``ctx`` persistence, no
    interaction with the resume state machine).

    Singleton-skip note: the cache is built unconditionally at layer
    entry. If every group in a layer is a singleton (``len(members) <= 1``
    when ``expert_distill_skip_singletons=True``), the cache build is
    wasted work — but the cost is a single softmax + topk + linear over
    ≤80 MB of fp32, while the simpler control flow is worth more than
    the all-singletons micro-saving (singletons are rare at production
    ``num_groups``).
    """
    # Step 1: deterministic sub-sample of ``layer_inputs`` — lifted
    # verbatim from ``_distill_merged_group`` (M-2 caveat: seed is
    # ``layer_idx`` ONLY, no global seed mix-in, so every centroid in
    # the same layer sees the SAME ``token_cap`` rows — that bias is
    # intentional for resume-determinism). This block is the
    # bit-identical twin of the legacy ``v2_router_cache is None`` branch
    # in ``_distill_merged_group``.
    rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
    n_tokens = layer_inputs.shape[0]
    if n_tokens > token_cap:
        idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
        x_all = layer_inputs[idx]
    else:
        x_all = layer_inputs
    x_all = x_all.to(device, dtype=torch.float32)

    # Step 2: full pre-merge router softmax + TopK-masked renormalization
    # (paper Eq. 2). The math here is the bit-identical twin of the v2
    # branch's per-group recompute at the now-deleted ``sigma =
    # _router_routing_weights(...)`` site — same call site, same
    # downstream mask + renormalize math, just hoisted up one frame so it
    # runs once per layer instead of once per centroid.
    with torch.no_grad():
        n_experts = layer_ref.num_routed_experts
        router_top_k = min(layer_ref.top_k, n_experts)
        sigma = _router_routing_weights(layer_ref, x_all)  # (T, n_experts)
        topk_idx = torch.topk(sigma, k=router_top_k, dim=-1).indices  # (T, k)
        topk_mask = torch.zeros_like(sigma, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        gate = sigma * topk_mask.to(sigma.dtype)  # (T, n_experts)
        gate_sum = gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gate = gate / gate_sum  # renormalized over top-k per token

    return {"x_all": x_all, "sigma": sigma, "gate": gate}


def _snapshot_pre_merge_layer_experts(
    layer_ref: MoELayerRef,
    members: "set[int] | None" = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """CPU snapshot of expert gate/up/down weights for a single layer,
    taken BEFORE the merge step mutates the bank.

    Used by step 7b (distillation) to compute the pre-merge group-member
    forward as the distillation target. Released by the per-layer driver
    once distillation finishes for the layer.

    When ``members`` is ``None`` (the default) ALL ``num_routed_experts``
    experts are snapshotted — preserving the legacy full-snapshot behavior
    relied on by direct callers/tests. When ``members`` is provided (the
    plugin path, computed as ``set().union(*grouped.values())``) only those
    expert ids are snapshotted: ``_distill_merged_group`` reads only group
    members (``pre_merge_weights[m]`` for ``m in members``), so non-member
    entries are dead weight in host RAM and narrowing them away is
    byte-identical for every distilled output.
    """
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}
    eids = range(layer_ref.num_routed_experts) if members is None else sorted(members)
    for eid in eids:
        out[eid] = {
            name: banks[name].get(eid).detach().cpu().clone()
            for name in MATRIX_NAMES
        }
    return out


def _distill_merged_group(
    *,
    layer_ref: MoELayerRef,
    centroid_id: int,
    members: list[int],
    freq: dict[int, int],
    pre_merge_weights: dict[int, dict[str, torch.Tensor]],
    layer_inputs: torch.Tensor,
    steps: int,
    lr: float,
    betas: tuple[float, float],
    plateau_steps: int,
    plateau_eps: float,
    token_cap: int,
    device: torch.device,
    use_ce_term: bool = False,
    ce_lambda: float = 1.0,
    target_version: str = "v1",
    v2_router_cache: dict | None = None,
) -> dict:
    """500-step MSE (+ optional CE) distillation of the merged centroid
    against the routing-gated pre-merge group-member forward
    (spec § 5 step 7b / M8).

    Target version (D-expert-distill-paper-lift):
    - ``target_version="v1"`` (helper default for back-compat): legacy
      freq-weighted-only target ``Σ (freq_e / Σ freq) · E_e^orig(x)``;
      drops the TopK gate AND the per-token routing weight
      ``g_e^orig(x)`` (see ``D-expert-distill-mse-v1`` below).
    - ``target_version="v2"`` (plugin __init__ default post-lift): the
      paper-faithful target from Eqs. 1-3 (paper lines 133-152),
      ``target = Σ_{e ∈ members ∩ TopK(σ_orig(x))} g_e^orig(x) · E_e^orig(x)``,
      where ``σ_orig`` is recomputed from ``layer_ref.router`` (the
      pre-merge router; the post-resize router carries the centroid's
      row verbatim so ``σ_orig`` on the original router gives the same
      ``Gate(x)`` distribution per Eq. 2). ``TopK`` is
      ``layer_ref.top_k``; the post-top-k softmax is renormalized per
      paper Eq. 2 ("TopK(X)_i = −∞ otherwise", paper line 145, which
      makes the softmax denominator the top-k sum). Tokens whose
      group has no top-k members contribute zero target.

    v2 still uses the reservoir-sampled layer-input ``layer_inputs``
    for the token pool (not the routing-restricted ``X_g`` set, which
    is impractical to reconstruct cheaply) — but the TopK mask makes
    that pool implicit per token: tokens not routed to any group
    member contribute zero target, which is the paper-correct
    semantic.

    Lift 1 — CE term (D-expert-distill-ce-term, paper Eq. 10)
    --------------------------------------------------------
    When ``use_ce_term=True`` (default OFF at this helper for back-compat;
    production default ON via the plugin __init__), the per-step loss is
        ``loss = L_CE + ce_lambda · MSE(student, target)``
    where ``L_CE`` is a per-token soft-KL between
    ``softmax(target)`` and ``log_softmax(student)`` over the hidden
    dim — the per-layer adaptation of MoE-Pruner's Eq. 10
    ``L_KD = L_CE + λ · L_expert`` (paper line 394; "LCE is the cross
    entropy loss" line 410). The paper's ``L_CE`` is the LM next-token
    cross-entropy from the end-to-end fine-tune; this distillation runs
    PER-LAYER PER-GROUP with no LM-head / labels in scope, so a
    feature-level KL on the same MSE-scope tokens is the closest faithful
    adaptation: the teacher signal is the pre-merge group-member forward
    (the same ``target`` the MSE term uses, already exposed via
    ``pre_merge_weights`` in ctx). Documented as a deviation rather than
    a hidden simplification — see ``D-expert-distill-ce-term`` in the
    module docstring.

    ``ce_lambda`` defaults to 1.0; the paper says "λ is a weighting
    coefficient and initialized based on the strength of cross entropy
    loss and expert-wise knowledge distillation loss" (line 414) without
    pinning a numeric default. The 1.0 default preserves the MSE
    magnitude verbatim and weights CE at parity — tune via config.

    Returns a small state dict with the final loss, step count, and break
    reason. The optimizer state is NOT persisted — resume re-runs the
    distillation from scratch for any layer whose partial JSON is missing.
    """
    # Pattern C config-validation: reject unknown target_version
    # versions at the helper boundary BEFORE any state setup so a
    # typo in config fails fast (also guards direct helper callers
    # in addition to the plugin-level validation). Runs before the
    # trivial-skip gate so that bogus values surface even when the
    # helper would have early-exited.
    if target_version not in ("v1", "v2"):
        raise ValueError(
            f"_distill_merged_group: target_version={target_version!r}; "
            "must be 'v1' or 'v2'."
        )

    if steps <= 0 or len(members) <= 1:
        return {"steps": 0, "skip": "trivial"}

    banks = build_banks(layer_ref)
    # Trainable: only the merged centroid's three projections. We pull the
    # current (post-merge) weights, wrap them as nn.Parameter, optimize, then
    # write back. Using nn.Parameter (not the bank tensors directly) lets us
    # build an optimizer cleanly without monkey-patching requires_grad on the
    # shared bank tensor.
    init_gate = banks["gate_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    init_up   = banks["up_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    init_down = banks["down_proj"].get(centroid_id).to(device, dtype=torch.float32).clone()
    p_gate = nn.Parameter(init_gate)
    p_up   = nn.Parameter(init_up)
    p_down = nn.Parameter(init_down)

    optim = torch.optim.AdamW(
        [p_gate, p_up, p_down], lr=lr, betas=betas, weight_decay=0.0,
    )

    # Token cap: subsample deterministically per layer for reproducibility.
    #
    # M-2 caveat: the reservoir feeding ``layer_inputs`` was itself
    # reservoir-sampled at profile time. Re-seeding here with
    # ``layer_idx`` ONLY (no global seed mix-in, no group / centroid
    # information) produces a deterministic second sub-sample that is
    # *not* a fresh random draw from the reservoir — every group in
    # the same layer trains on the same ``token_cap`` rows. That is
    # intentional for resume-determinism, but it does bias the training
    # pool: rare-token coverage hinges on whichever ``token_cap``
    # indices ``randperm(layer_idx)`` happens to pick. A future
    # randomized-sub-sampling pass (e.g., seed mixed with centroid id +
    # step) would decouple this from the reservoir's draw without
    # breaking resume.
    #
    # E-2 fix: when ``v2_router_cache`` is provided (plugin path), reuse
    # the cached ``x_all`` directly — it is bit-identical to what this
    # legacy branch would produce (same ``layer_idx`` seed, same cap, same
    # cast). Direct helper callers (tests, ad-hoc usage) pass
    # ``v2_router_cache=None`` and keep the legacy in-helper sub-sample
    # path verbatim for byte-identity.
    if v2_router_cache is not None:
        x_all = v2_router_cache["x_all"]
    else:
        rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
        n_tokens = layer_inputs.shape[0]
        if n_tokens > token_cap:
            idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
            x_all = layer_inputs[idx]
        else:
            x_all = layer_inputs
        x_all = x_all.to(device, dtype=torch.float32)

    # Build the per-token distillation target once (it doesn't change
    # during training because the pre-merge expert weights and the
    # pre-merge router are both frozen w.r.t. the optimizer).
    if target_version == "v1":
        # D-expert-distill-mse-v1: freq-weighted average of pre-merge
        # member outputs on the same token pool — drops TopK gate and
        # per-token routing weight. Preserved for ablation parity.
        weights = np.array(
            [max(freq.get(m, 0), 0) for m in members], dtype=np.float64,
        )
        if weights.sum() <= 0.0:
            weights[:] = 1.0
        weights = weights / weights.sum()

        with torch.no_grad():
            target = torch.zeros_like(x_all)
            for w, m in zip(weights, members):
                W_g = pre_merge_weights[m]["gate_proj"].to(device, dtype=torch.float32)
                W_u = pre_merge_weights[m]["up_proj"  ].to(device, dtype=torch.float32)
                W_d = pre_merge_weights[m]["down_proj"].to(device, dtype=torch.float32)
                target = target + float(w) * _swiglu_forward(W_g, W_u, W_d, x_all)
    else:
        # target_version == "v2": D-expert-distill-paper-lift. Paper
        # Eqs. 1-3 (lines 133-152): ``y = Σ_i Gate(x)_i · E_i(x)`` with
        # ``Gate(x) = Softmax(TopK(x · W_g))``. Applied to the
        # per-merge-group additive target, we sum only over group
        # members that fall inside the per-token TopK and weight each
        # by the renormalized post-top-k softmax mass.
        with torch.no_grad():
            # Full-softmax routing weights ``σ_orig(x)`` from the
            # pre-merge router. Returned as (T, n_experts) float32.
            # Note: ``_router_routing_weights`` returns the UNMASKED
            # softmax (matching the cost-builder convention); we apply
            # the TopK mask + renormalization below to match paper
            # Eq. 2's "Softmax(TopK(...))" verbatim. The output-space
            # cost builder intentionally uses the un-renormalized form
            # (see ``D-output-space-routing-weight``); the paper's
            # distillation loss explicitly states Eq. 2's renormalized
            # form, so v2 follows the paper here.
            #
            # E-2 fix: when ``v2_router_cache`` is provided (plugin
            # path), reuse the pre-computed ``sigma`` + ``gate``
            # directly — both are centroid-independent and the cache
            # hoists the computation to layer entry (one call per
            # layer instead of ``len(grouped)``). Direct helper callers
            # without the cache hit the in-helper recompute below.
            if v2_router_cache is not None:
                sigma = v2_router_cache["sigma"]
                gate = v2_router_cache["gate"]
            else:
                n_experts = layer_ref.num_routed_experts
                router_top_k = min(layer_ref.top_k, n_experts)
                sigma = _router_routing_weights(layer_ref, x_all)  # (T, n_experts)
                # Paper Eq. 2 (line 144-145): "TopK(X)_i = l_i if i is in the
                # top-K coordinates of logits l and TopK(X)_i = −∞
                # otherwise." → mask logits with -∞ outside top-k, then
                # softmax. Equivalent (and numerically friendlier) form:
                # zero-out non-top-k softmax probs and renormalize to sum
                # to 1 across top-k.
                # math: softmax(logit; mask=−∞) ≡ softmax_full × topk_mask / sum
                # (the −∞ entries vanish under exp(); the surviving top-k
                # entries normalize to the same renormalized softmax we get
                # by masking-then-dividing the full softmax — tying this
                # implementation directly to paper Eq. 2's mask form.)
                topk_idx = torch.topk(sigma, k=router_top_k, dim=-1).indices  # (T, k)
                topk_mask = torch.zeros_like(sigma, dtype=torch.bool)
                topk_mask.scatter_(1, topk_idx, True)
                gate = sigma * topk_mask.to(sigma.dtype)  # (T, n_experts)
                gate_sum = gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                gate = gate / gate_sum  # renormalized over top-k per token

            target = torch.zeros_like(x_all)
            for m in members:
                # Per-member gate column g_m^orig(x) — already zero on
                # tokens where m is not in the top-k (mask applied
                # above). Tokens routed through m contribute the
                # renormalized softmax weight × E_m's SwiGLU output.
                g_m = gate[:, m].unsqueeze(-1)  # (T, 1)
                W_g = pre_merge_weights[m]["gate_proj"].to(device, dtype=torch.float32)
                W_u = pre_merge_weights[m]["up_proj"  ].to(device, dtype=torch.float32)
                W_d = pre_merge_weights[m]["down_proj"].to(device, dtype=torch.float32)
                E_m = _swiglu_forward(W_g, W_u, W_d, x_all)  # (T, hidden)
                target = target + g_m * E_m

    # v2 student-gate: the spec target form is
    # ``student = g_g^merged(x) · E_g^merged(x)`` (D-expert-distill-mse,
    # module docstring) — symmetric to the v2 target. Under v2 we
    # multiply the student SwiGLU output by the centroid's per-token
    # gate weight from the SAME pre-merge router (the post-resize
    # router carries the centroid's pre-merge row verbatim, so this
    # IS the post-resize centroid's gate up to the denominator
    # change Stage 2.5 retrains). Under v1 the student stays un-gated
    # to preserve byte-identity with the legacy path.
    if target_version == "v2":
        student_gate = gate[:, centroid_id].unsqueeze(-1)  # (T, 1) — no_grad above
    else:
        student_gate = None

    def _student_forward() -> torch.Tensor:
        out = _swiglu_forward(p_gate, p_up, p_down, x_all)
        if student_gate is not None:
            out = student_gate * out
        return out

    # LOW-1 fix: snapshot the *pre-step-0* loss as the relative-loss
    # baseline. Previously this was set inside the loop after step 0's
    # update, which meant "relative_loss = final / initial" measured
    # progress against a baseline that already incorporated one AdamW
    # update — optimistically biased. The fp32 no-grad forward below is
    # the true distillation starting point (post-merge centroid vs.
    # freq-weighted target on the same token batch).
    #
    # Lift 1: when ``use_ce_term=True`` the baseline includes the CE
    # term too so ``relative_loss = final / initial`` measures the same
    # composite quantity the optimizer is actually minimizing. Plateau
    # threshold semantics are unchanged.
    with torch.no_grad():
        student0 = _student_forward()
        mse0 = F.mse_loss(student0, target)
        if use_ce_term:
            ce0 = _feature_kl_ce(student0, target)
            loss0 = ce0 + ce_lambda * mse0
        else:
            loss0 = mse0
        initial_loss = max(float(loss0.item()), 1e-12)

    plateau_counter = 0
    last_step = 0
    final_loss = float(initial_loss)
    break_reason = "max_steps"

    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        student = _student_forward()
        mse_loss = F.mse_loss(student, target)
        if use_ce_term:
            # Paper Eq. 10: ``L_KD = L_CE + λ · L_expert`` (line 394).
            # ``L_expert`` is the per-expert MSE (the ``mse_loss`` above
            # against the freq-weighted / TopK-gated target per the
            # active v1/v2 path); ``L_CE`` is the paper's cross-entropy
            # term, adapted to the per-layer per-group scope as a
            # feature-level KL (see ``D-expert-distill-ce-term``). The
            # arithmetic combination follows the paper's formula verbatim.
            ce_loss = _feature_kl_ce(student, target)
            loss = ce_loss + ce_lambda * mse_loss
        else:
            loss = mse_loss
        loss.backward()
        optim.step()
        last_step = step + 1
        final_loss = float(loss.detach().item())

        # Plateau early-break: ``relative_loss = final / initial`` falling
        # below ``plateau_eps`` for ``plateau_steps`` consecutive steps stops
        # training. Uses < (strict) so the very first step at exact threshold
        # is NOT counted, matching spec wording "below 1e-4 of the initial".
        if final_loss / initial_loss < plateau_eps:
            plateau_counter += 1
            if plateau_counter >= plateau_steps:
                break_reason = "plateau"
                break
        else:
            plateau_counter = 0

    # Write the trained weights back to the bank in the original dtype.
    #
    # Resume disclaimer (LOW-3): we train in fp32 but cast the result
    # back to the bank dtype (typically bf16) on writeback. The
    # optimizer state is NOT persisted (see the docstring above) — a
    # resume that re-enters this loop after a crash will:
    #   1. read the bank's bf16 centroid as the new fp32 init, and
    #   2. start AdamW from that bf16-rounded point, not from the
    #      pre-crash fp32 trajectory.
    # That is intentional (no optimizer-state on disk) but worth noting:
    # the resumed trajectory is NOT bit-equal to the non-crash
    # trajectory because of the bf16 round-trip at the resume boundary.
    bank_dtype = banks["gate_proj"].get(centroid_id).dtype
    with torch.no_grad():
        banks["gate_proj"].set(centroid_id, p_gate.detach().to(bank_dtype))
        banks["up_proj"  ].set(centroid_id, p_up.detach().to(bank_dtype))
        banks["down_proj"].set(centroid_id, p_down.detach().to(bank_dtype))

    return {
        "steps": last_step,
        "final_loss": final_loss,
        "initial_loss": float(initial_loss) if initial_loss is not None else None,
        "break_reason": break_reason,
    }


class ExpertDistillPlugin:
    """Plugin home for Stage 2 per-merge-group expert distillation
    (spec § 5 step 7b / M8).

    LIVE as of S2-11: this plugin owns the per-merge-group expert distillation
    on the decomposed phase walk. ``pre_merge_snapshot`` snapshots every
    expert's weights BEFORE the merge mutates the bank; ``merge`` runs the
    ``_distill_merged_group`` loop AFTER ``LegacyAdapter`` has done
    ``_merge_experts_inplace`` — the orchestrator registers this plugin after
    the adapter so the phase-major / plugin-minor walk lands its ``merge`` hook
    after the adapter's. The distillation MUST run in the ``merge`` phase
    (between ``_merge_experts_inplace`` and ``bank.select``), NOT ``post_merge``.

    Config gate: enabled iff ``stage2_reap_ream.expert_distill_steps`` is a
    positive integer. ``expert_distill_steps`` is a numeric knob (default 0).
    """

    name = "expert_distill"
    paper = (
        "Per-merge-group MSE (+ optional CE) distillation against "
        "routing-gated original outputs. Inspired by MoE-Pruner "
        "arXiv:2410.12013 (expert-wise MSE, Eq. 10; paper-cited repo "
        "github.com/yanyue-xie/moe-pruner is 404 — closest third-party "
        "re-implementation is github.com/tanganke/fusion_bench under "
        "fusion_bench/method/moe_pruner/, but it ships only the pruning "
        "pass and NO distillation code, so Eq. 10 has no public reference "
        "implementation) and SlimMoE arXiv:2506.18349 (top-8 logits KL "
        "on full model output, Eq. 3) — no project-aligned per-merge-group "
        "code in either. REAM baseline arXiv:2604.04356 has no post-merge "
        "distill. Deviations: D-expert-distill-mse (per-group additive "
        "target against pre-merge members; expert/router training "
        "separated), D-expert-distill-mse-v1 (freq-weighted target + "
        "reservoir tokens — drops TopK gate AND per-token routing weight), "
        "D-expert-distill-ce-term (Lift 1: paper Eq. 10's L_CE term "
        "adapted to per-layer per-group as a feature-level KL between "
        "pre-merge teacher softmax and merged-student log-softmax — "
        "default ON), D-expert-distill-paper-lift (Lift 2: v2 target "
        "with TopK gate + per-token routing weight per Eqs. 1-3 — "
        "default 'v2'; v1 retained for ablation parity). See module "
        "docstring."
    )
    config_key = "stage2_reap_ream.expert_distill_steps"
    # S2-11 LIVE: pre_merge_snapshot reads layer_ref and writes
    # pre_merge_weights; merge reads the merge-group state + accumulators and
    # overwrites distill_state.
    reads: tuple[str, ...] = (
        "layer_ref", "pre_merge_weights", "grouped", "freq", "layer_input_acc",
    )
    writes: tuple[str, ...] = ("pre_merge_weights", "distill_state")
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        expert_distill_steps: int,
        expert_distill_lr: float,
        expert_distill_betas: tuple[float, float],
        expert_distill_token_cap: int,
        expert_distill_skip_singletons: bool,
        expert_distill_plateau_steps: int,
        expert_distill_plateau_eps: float,
        expert_distill_use_ce_term: bool = True,
        expert_distill_ce_lambda: float = 1.0,
        expert_distill_target_version: str = "v2",
    ) -> None:
        """Store every distill knob the live hooks read.

        The knob set mirrors the ``expert_distill_*`` block of
        ``LegacyAdapter.__init__`` exactly — no logic in ``__init__``, just a
        faithful re-host of the local variables the distill code read off
        ``self`` in the pre-S2-11 adapter.

        Lift 1 additions (D-expert-distill-ce-term):
        - ``expert_distill_use_ce_term`` (default True): when True,
          per-step loss is ``feature_KL + ce_lambda · MSE``. NOTE: the
          ``feature_KL`` term is the in-plugin engineering adaptation
          (``_feature_kl_ce``), NOT paper-faithful Eq. 10 ``L_CE`` —
          see the module docstring's "RESOLVED" block for the rationale
          (paper-literal vocab CE was rejected as a ~500×/step
          regression; the paper-faithful term is deferred to a downstream
          fine-tune phase). When False, runs pure MSE (v1 back-compat /
          A0..A11 ablation parity). Pattern C config-validation: the
          knob is consumed verbatim, no implicit coupling.
        - ``expert_distill_ce_lambda`` (default 1.0): λ-parity weight
          between the feature-KL term and the MSE term — NOT paper
          Eq. 10's λ (which weights the deferred vocab-level ``L_CE``,
          a different scale entirely). 1.0 is the safe parity default
          for the in-plugin feature-KL adaptation; setting it to 0
          falls back to MSE-only behavior. If/when the downstream
          paper-faithful ``L_CE`` fine-tune phase lands, expect that
          phase to pin its own λ-equivalent against vocab CE
          magnitudes, independent of this knob.

        Lift 2 addition (D-expert-distill-paper-lift):
        - ``expert_distill_target_version``: ``"v1"`` (legacy
          freq-weighted target) or ``"v2"`` (default; paper-faithful
          TopK-gated + per-token routing-weighted target per Eqs.
          1-3, paper lines 133-152). v1 is retained for A0..A11
          ablation parity and back-compat. Pattern C: validated at
          this boundary so a typo cannot silently fall through.
        """
        if expert_distill_target_version not in ("v1", "v2"):
            raise ValueError(
                "expert_distill_target_version="
                f"{expert_distill_target_version!r}; must be "
                "'v1' (legacy freq-weighted target) or 'v2' "
                "(paper-faithful TopK-gated target — D-expert-distill"
                "-paper-lift)."
            )
        self.expert_distill_steps = expert_distill_steps
        self.expert_distill_lr = expert_distill_lr
        self.expert_distill_betas = expert_distill_betas
        self.expert_distill_token_cap = expert_distill_token_cap
        self.expert_distill_skip_singletons = expert_distill_skip_singletons
        self.expert_distill_plateau_steps = expert_distill_plateau_steps
        self.expert_distill_plateau_eps = expert_distill_plateau_eps
        self.expert_distill_use_ce_term = expert_distill_use_ce_term
        self.expert_distill_ce_lambda = expert_distill_ce_lambda
        self.expert_distill_target_version = expert_distill_target_version

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.expert_distill_steps`` > 0.

        Defaults to 0 (distillation off) → a missing key / block leaves the
        plugin disabled. Coerced via ``int(...)`` to match the
        ``steps <= 0`` guard inside ``_distill_merged_group``; a non-numeric
        value falls back to disabled rather than crashing config discovery.
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        try:
            return int(s2.get("expert_distill_steps", 0)) > 0
        except (TypeError, ValueError):
            return False

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def pre_merge_snapshot(self, ctx: PipelineContext) -> None:
        """Snapshot pre-merge expert weights for distillation (LIVE S2-11).

        Verbatim lift of the distill-snapshot part of
        ``LegacyAdapter.pre_merge_snapshot``: snapshot every expert's gate/up/
        down weights BEFORE ``_merge_experts_inplace`` mutates the bank, so the
        per-group distillation step in ``merge`` can compute the pre-merge
        group-member forward as the self-distillation target. Snapshots only
        when distillation is enabled (``expert_distill_steps > 0``) — keeps
        host-RAM cost zero for disabled runs. Writes the ``pre_merge_weights``
        ctx slot (``None`` when disabled).
        """
        layer_ref = ctx.get("layer_ref")
        # Phase 3 (M8): snapshot pre-merge expert weights BEFORE the merge
        # mutates the bank. The snapshot is consumed only by the per-group
        # distillation step in ``merge``; released as soon as that finishes
        # for this layer (Python GC since no module-level reference is held).
        #
        # Lever 4: narrow the snapshot to only the experts any merge group can
        # read (``set().union(*grouped.values())`` = every centroid + every
        # absorbed/promoted member). ``_distill_merged_group`` reads only
        # ``pre_merge_weights[m]`` for ``m in members``, so the narrowed dict is
        # an exact superset of every key consumed — byte-identical distilled
        # output, less host RAM. The ``if grouped`` guard degrades to a full
        # snapshot if ``grouped`` is empty/None (defensive; not expected live).
        # ``grouped`` is only needed when distillation is enabled; on the live
        # path it is ``set`` (orchestrator:608, inside _run_assignment) strictly
        # before this post-assign phase fires (orchestrator:1568). Read it via
        # ``has`` so the disabled path (which sets only ``layer_ref``) never
        # raises a KeyError.
        if self.expert_distill_steps > 0:
            grouped = ctx.get("grouped") if ctx.has("grouped") else None
            needed = set().union(*grouped.values()) if grouped else None
            pre_merge_weights: dict[int, dict[str, torch.Tensor]] | None = (
                _snapshot_pre_merge_layer_experts(layer_ref, members=needed)
            )
        else:
            pre_merge_weights = None
        ctx.set("pre_merge_weights", pre_merge_weights)

    def merge(self, ctx: PipelineContext) -> None:
        """Per-merge-group expert distillation (LIVE S2-11).

        Verbatim lift of the distillation block from ``LegacyAdapter.merge``
        (the per-group ``_distill_merged_group`` loop). Runs in the ``merge``
        phase AFTER ``LegacyAdapter._merge_experts_inplace`` (the orchestrator
        registers this plugin after the adapter) and BEFORE ``bank.select`` in
        ``LegacyAdapter.post_merge``. Overwrites the ``distill_state`` ctx slot
        (``LegacyAdapter.merge`` sets it to ``None`` as a default first).
        """
        layer_ref = ctx.get("layer_ref")
        grouped = ctx.get("grouped")
        freq = ctx.get("freq")
        layer_input_acc = ctx.get("layer_input_acc")
        pre_merge_weights = ctx.get("pre_merge_weights")

        # Phase 3 (M8): per-merge-group expert distillation (spec § 5 step 7b).
        distill_state: dict[int, dict] | None = None
        if self.expert_distill_steps > 0 and pre_merge_weights is not None:
            layer_inputs_buf = (
                layer_input_acc.get() if layer_input_acc is not None else None
            )
            if layer_inputs_buf is None or layer_inputs_buf.shape[0] == 0:
                log.warning(
                    "layer %d: expert distillation enabled but no layer-input "
                    "samples were captured during profile — skipping.",
                    layer_ref.layer_idx,
                )
            else:
                distill_state = {}
                target_device = layer_ref.layer_module.parameters().__next__().device
                # E-2: build the per-layer router cache ONCE for v2 — all
                # centroid groups in this layer share the same
                # ``(x_all, sigma, gate)`` triple (the pre-merge router
                # and the layer's reservoir-sub-sampled tokens are both
                # frozen for the whole ``merge`` invocation). For v1
                # the cache is not used (v1 never calls
                # ``_router_routing_weights``); we still pass
                # ``v2_router_cache=None`` for that path, so the helper
                # falls back to its legacy in-helper sub-sample. Plan
                # OQ #1: build unconditionally even when every group is
                # a singleton — the build is one softmax+topk+linear
                # (≤80 MB fp32) and the simpler control flow beats the
                # all-singletons micro-saving.
                v2_router_cache: dict | None = None
                if self.expert_distill_target_version == "v2":
                    v2_router_cache = _build_v2_router_cache(
                        layer_ref=layer_ref,
                        layer_inputs=layer_inputs_buf,
                        token_cap=self.expert_distill_token_cap,
                        device=target_device,
                    )
                for centroid, members in grouped.items():
                    if self.expert_distill_skip_singletons and len(members) <= 1:
                        continue
                    state = _distill_merged_group(
                        layer_ref=layer_ref,
                        centroid_id=centroid,
                        members=members,
                        freq=freq,
                        pre_merge_weights=pre_merge_weights,
                        layer_inputs=layer_inputs_buf,
                        steps=self.expert_distill_steps,
                        lr=self.expert_distill_lr,
                        betas=self.expert_distill_betas,
                        plateau_steps=self.expert_distill_plateau_steps,
                        plateau_eps=self.expert_distill_plateau_eps,
                        token_cap=self.expert_distill_token_cap,
                        device=target_device,
                        use_ce_term=self.expert_distill_use_ce_term,
                        ce_lambda=self.expert_distill_ce_lambda,
                        target_version=self.expert_distill_target_version,
                        v2_router_cache=v2_router_cache,
                    )
                    distill_state[centroid] = state
                log.info(
                    "  layer %d distillation: %d non-singleton groups distilled",
                    layer_ref.layer_idx, len(distill_state),
                )

        ctx.set("distill_state", distill_state, overwrite=True)
