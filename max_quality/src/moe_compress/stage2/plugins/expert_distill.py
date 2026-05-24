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

Official code
-------------
MoE-Pruner has official code at github.com/yanyue-xie/moe-pruner, but
no project-aligned per-merge-group implementation exists: their
expert-wise loss pairs teacher expert ``i`` with student expert ``i``
across the unpruned/pruned bank, whereas this plugin pairs the merged
centroid with the *additive contribution of its pre-merge group
members*. The ``_distill_merged_group`` loop is project-original.

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
from .output_space_cost import _swiglu_forward

log = logging.getLogger(__name__)


# ===========================================================================
# Phase 3 — per-merge-group expert distillation (spec § 5 step 7b / M8)
# ===========================================================================


def _snapshot_pre_merge_layer_experts(
    layer_ref: MoELayerRef,
) -> dict[int, dict[str, torch.Tensor]]:
    """CPU snapshot of every expert's gate/up/down weights for a single
    layer, taken BEFORE the merge step mutates the bank.

    Used by step 7b (distillation) to compute the pre-merge group-member
    forward as the distillation target. Released by the per-layer driver
    once distillation finishes for the layer.

    v1-waste note: this snapshots ALL ``num_routed_experts`` experts for
    the layer, but ``_distill_merged_group`` only reads members of each
    merge group (``pre_merge_weights[m]`` for ``m in members``). The
    spec-form v2 target only needs the group's members, so the
    non-member entries are dead weight in host RAM. v2 should narrow
    the snapshot to ``set().union(*grouped.values())`` once ``grouped``
    is available pre-merge.
    """
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}
    n = layer_ref.num_routed_experts
    for eid in range(n):
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
) -> dict:
    """500-step MSE distillation of the merged centroid against the
    freq-weighted pre-merge group-member forward (spec § 5 step 7b / M8).

    **v1 simplification — see D-expert-distill-mse-v1 in spec § 10**: this
    implementation differs from the pinned spec target in two ways:
    (i) freq-weighted-only target ``Σ (freq_e / Σ freq) · E_e^orig(x)``
        (no per-token routing weight ``g_e^orig(x)``);
    (ii) input tokens are the reservoir-sampled layer-input ``layer_inputs``
         for every group, not the routing-restricted ``X_g`` set.
    Phase 3 v2 will lift both. The v1 form provides a correctly-signed
    merge-error gradient on a uniform-token sample.

    Returns a small state dict with the final loss, step count, and break
    reason. The optimizer state is NOT persisted — resume re-runs the
    distillation from scratch for any layer whose partial JSON is missing.
    """
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
    rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
    n_tokens = layer_inputs.shape[0]
    if n_tokens > token_cap:
        idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
        x_all = layer_inputs[idx]
    else:
        x_all = layer_inputs
    x_all = x_all.to(device, dtype=torch.float32)

    # Build the freq-weighted target once (it doesn't change during training).
    weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
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

    # LOW-1 fix: snapshot the *pre-step-0* loss as the relative-loss
    # baseline. Previously this was set inside the loop after step 0's
    # update, which meant "relative_loss = final / initial" measured
    # progress against a baseline that already incorporated one AdamW
    # update — optimistically biased. The fp32 no-grad forward below is
    # the true distillation starting point (post-merge centroid vs.
    # freq-weighted target on the same token batch).
    with torch.no_grad():
        student0 = _swiglu_forward(p_gate, p_up, p_down, x_all)
        initial_loss = max(float(F.mse_loss(student0, target).item()), 1e-12)

    plateau_counter = 0
    last_step = 0
    final_loss = float(initial_loss)
    break_reason = "max_steps"

    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        student = _swiglu_forward(p_gate, p_up, p_down, x_all)
        loss = F.mse_loss(student, target)
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
        "Per-merge-group MSE distillation against routing-gated original "
        "outputs. Inspired by MoE-Pruner arXiv:2410.12013 (expert-wise MSE, "
        "Eq. 10; official code at github.com/yanyue-xie/moe-pruner) and "
        "SlimMoE arXiv:2506.18349 (top-8 logits KL on full model output, "
        "Eq. 3) — no project-aligned per-merge-group code in either. REAM "
        "baseline arXiv:2604.04356 has no post-merge distill. Deviations: "
        "D-expert-distill-mse (per-group additive target against pre-merge "
        "members; expert/router training separated), D-expert-distill-mse-v1 "
        "(freq-weighted target + reservoir tokens — drops TopK gate AND "
        "per-token routing weight). See module docstring."
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
    ) -> None:
        """Store every distill knob the live hooks read.

        The knob set mirrors the ``expert_distill_*`` block of
        ``LegacyAdapter.__init__`` exactly — no logic in ``__init__``, just a
        faithful re-host of the local variables the distill code read off
        ``self`` in the pre-S2-11 adapter.
        """
        self.expert_distill_steps = expert_distill_steps
        self.expert_distill_lr = expert_distill_lr
        self.expert_distill_betas = expert_distill_betas
        self.expert_distill_token_cap = expert_distill_token_cap
        self.expert_distill_skip_singletons = expert_distill_skip_singletons
        self.expert_distill_plateau_steps = expert_distill_plateau_steps
        self.expert_distill_plateau_eps = expert_distill_plateau_eps

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
        pre_merge_weights: dict[int, dict[str, torch.Tensor]] | None = (
            _snapshot_pre_merge_layer_experts(layer_ref)
            if self.expert_distill_steps > 0
            else None
        )
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
                    )
                    distill_state[centroid] = state
                log.info(
                    "  layer %d distillation: %d non-singleton groups distilled",
                    layer_ref.layer_idx, len(distill_state),
                )

        ctx.set("distill_state", distill_state, overwrite=True)
