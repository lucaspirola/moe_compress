"""Per-merge-group expert distillation (Task 16 of the plugin-architecture refactor).

Home of ``_distill_merged_group`` — the Phase 3 / step-7b MSE distillation
trainer that fine-tunes each merged centroid against the freq-weighted
pre-merge group-member forward — and its helper
``_snapshot_pre_merge_layer_experts``, which CPU-snapshots every expert's
weights before the merge mutates the bank. Both moved verbatim out of
``stage2_reap_ream.py``; that module re-imports them so external callers and
tests keep their existing import paths.

Circular-import note: this module imports only ``moe_compress.utils.model_io``,
``pipeline.base``, ``pipeline.context`` and ``pipeline.plugins.output_space_cost``
(for ``_swiglu_forward``) — none of which import ``stage2_reap_ream`` or
``expert_distill``. There is therefore no cycle at module load, and every
import below is a plain module-top import (no function-scope late imports).

``ExpertDistillPlugin`` is a scaffold-only plugin — not yet on the
live phase walk. ``LegacyAdapter.pre_merge_snapshot`` / ``LegacyAdapter.merge``
still call ``_snapshot_pre_merge_layer_experts`` / ``_distill_merged_group``
directly (via a late ``from ...stage2_reap_ream import ...``), and the
``MOE_STAGE2_LEGACY_LOOP=1`` path in ``stage2_reap_ream.run()`` does too. This
class gives T17/T18 a per-distiller plugin to wire into the decomposed
``pre_merge_snapshot`` / ``post_merge`` phases.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks
from ...pipeline.context import PipelineContext
from .output_space_cost import _swiglu_forward


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

    initial_loss = None
    plateau_counter = 0
    last_step = 0
    final_loss = float("inf")
    break_reason = "max_steps"

    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        student = _swiglu_forward(p_gate, p_up, p_down, x_all)
        loss = F.mse_loss(student, target)
        loss.backward()
        optim.step()
        last_step = step + 1
        final_loss = float(loss.detach().item())

        if initial_loss is None:
            initial_loss = max(final_loss, 1e-12)

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

    T16 status: scaffold only — NOT on the live phase walk.
    ``LegacyAdapter.pre_merge_snapshot`` still calls
    ``_snapshot_pre_merge_layer_experts`` and ``LegacyAdapter.merge`` still
    calls ``_distill_merged_group`` directly; the ``MOE_STAGE2_LEGACY_LOOP=1``
    path in ``stage2_reap_ream.run()`` does too. This class exists so T17/T18
    have a per-distiller plugin to wire into the decomposed
    ``pre_merge_snapshot`` / ``post_merge`` phases.

    Config gate: enabled iff ``stage2_reap_ream.expert_distill_steps`` is a
    positive integer. ``expert_distill_steps`` is a numeric knob (default 0).
    """

    name = "expert_distill"
    paper = "Per-merge-group expert distillation (spec § 5 step 7b / M8)."
    config_key = "stage2_reap_ream.expert_distill_steps"
    # () until a later task wires the live hook
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

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
        """Documented no-op for T16.

        The live pre-merge snapshot still belongs to
        ``LegacyAdapter.pre_merge_snapshot`` (and the
        ``MOE_STAGE2_LEGACY_LOOP=1`` path), which call
        ``_snapshot_pre_merge_layer_experts`` directly. Returning ``None``
        makes this hook a clean no-op. T17/T18 wire the real call here once
        ``pre_merge_snapshot`` is decomposed into the fine-grained phase walk.
        """
        return None

    def post_merge(self, ctx: PipelineContext) -> None:
        """Documented no-op for T16.

        The live per-group distillation still belongs to
        ``LegacyAdapter.merge`` (and the ``MOE_STAGE2_LEGACY_LOOP=1`` path),
        which call ``_distill_merged_group`` directly. Returning ``None``
        makes this hook a clean no-op. T17/T18 wire the real call here once
        the post-merge phase is decomposed.
        """
        return None
