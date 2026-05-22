"""Per-layer merge-heal by self-distillation (Task 17 of the plugin-architecture refactor).

Home of the Stage-2 merge-heal toolchain — all opt-in, all inert when
``merge_heal_enabled`` is False:
  * ``_HealConfig``            — validated, frozen view of the merge_heal_* knobs.
  * ``_make_shared_out_fn``    — frozen-shared-expert closure builder.
  * ``_capture_mlp_io``        — pre-merge (input, target) activation capture.
  * ``_heal_student_moe_output`` — faithful student replica of the MoE block.
  * ``_heal_lr_at_step``       — warmup -> cosine -> floor LR schedule.
  * ``_heal_layer``            — per-layer self-distillation trainer.
  * ``_summarize_distill_state`` — per-layer distill telemetry aggregator.

All seven moved verbatim out of ``stage2_reap_ream.py``; that module
re-imports them so external callers and tests keep their existing import
paths. ``_write_heal_weights`` / ``_load_heal_weights`` live in
``pipeline.shared_io`` (Task 2) and are NOT touched here — none of the seven
functions reference them.

After Stage 2 merges a layer, optionally fine-tune that layer's kept experts
(+ optionally its resized router) by SELF-DISTILLATION: the layer is trained
to reproduce its OWN pre-merge MoE-block output on calibration tokens. The
(input, target) pairs are captured in-process from the layer itself just
before the merge — no teacher object, no disk sidecar, no cascade buffer.
Every code path here is reached ONLY when ``merge_heal_enabled`` is True; with
the flag off, none of this runs and Stage 2 behaviour is byte-identical to the
pre-feature code.

Circular-import note: this module imports only ``moe_compress.utils.*``
(model_io / activation_shards / activation_hooks), ``pipeline.base``,
``pipeline.context`` and ``pipeline.plugins.output_space_cost`` (for
``_swiglu_forward``) — none of which import ``stage2_reap_ream`` or
``merge_heal``. There is therefore no cycle at module load, and every import
below is a plain module-top import (no function-scope late imports).

``MergeHealPlugin`` is a scaffold-only ``Stage2Plugin`` — not yet on the live
phase walk. ``LegacyAdapter.pre_merge_snapshot`` / ``post_merge`` /
``write_artifacts`` still call ``_capture_mlp_io`` / ``_make_shared_out_fn`` /
``_heal_layer`` / ``_summarize_distill_state`` directly (via a late
``from ...stage2_reap_ream import ...``), and the ``MOE_STAGE2_LEGACY_LOOP=1``
path in ``stage2_reap_ream.run()`` does too. This class gives T18 a per-layer
merge-heal plugin to wire into the decomposed phase walk.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.activation_hooks import _EarlyExitException, early_exit_after_layer
from ...utils.activation_shards import (
    HealActivationDataset,
    ShardManifest,
    ShardWriter,
)
from ...utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks
from .._framework.base import Stage2Plugin
from ...pipeline.context import PipelineContext
from .output_space_cost import _swiglu_forward

log = logging.getLogger(__name__)


class _HealConfig:
    """Validated, frozen view of the ``merge_heal_*`` config knobs.

    Built once at the top of :func:`run` from the ``stage2_reap_ream`` config
    block. All knobs default to OFF/inert values; ``enabled`` gates every
    merge-heal code path so a config that omits the block (or sets
    ``merge_heal_enabled: false``) reproduces the pre-feature behaviour.
    """

    __slots__ = (
        "enabled", "train_router", "lr", "adamw_betas", "grad_clip",
        "holdout_fraction", "patience", "eval_interval", "max_steps",
        "token_cap", "minibatch_size", "min_rel_delta",
        "lr_warmup_steps", "lr_min", "lr_decay_steps",
        "cross_domain_holdout_enabled", "xd_holdout_tokens",
        "shard_rows", "shard_dir", "keep_shards",
    )

    def __init__(self, s2: dict) -> None:
        self.enabled: bool = bool(s2.get("merge_heal_enabled", False))
        self.train_router: bool = bool(s2.get("merge_heal_train_router", True))
        self.lr: float = float(s2.get("merge_heal_lr", 1.0e-4))
        _betas = s2.get("merge_heal_adamw_betas", [0.9, 0.95])
        # This guard runs unconditionally (not gated on self.enabled) because
        # the (float(_betas[0]), float(_betas[1])) unpack below is itself
        # unconditional — a scalar or a string would otherwise crash there with
        # an opaque TypeError / produce float("0")-style nonsense.
        if not isinstance(_betas, (list, tuple)) or len(_betas) != 2:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_adamw_betas={_betas}; "
                "must be a length-2 list [beta1, beta2] of exactly two "
                "numeric values."
            )
        self.adamw_betas: tuple[float, float] = (float(_betas[0]), float(_betas[1]))
        self.grad_clip: float = float(s2.get("merge_heal_grad_clip", 1.0))
        self.holdout_fraction: float = float(s2.get("merge_heal_holdout_fraction", 0.10))
        self.patience: int = int(s2.get("merge_heal_patience", 5))
        self.eval_interval: int = int(s2.get("merge_heal_eval_interval", 25))
        self.max_steps: int = int(s2.get("merge_heal_max_steps", 2000))
        self.token_cap: int = int(s2.get("merge_heal_token_cap", 262144))
        self.minibatch_size: int = int(
            s2.get("merge_heal_minibatch_size", 8192)
        )
        # Minimum relative improvement that counts as "improvement" for the
        # patience early-stop. Without this (the historical 0.0), an asymptotic
        # curve is always microscopically descending — noise-level micro-gains
        # reset patience forever and every layer runs to max_steps. Setting
        # this > 0 means an eval must beat the running best by at least this
        # *fraction* to reset patience; the best snapshot still updates on
        # any strict improvement, so the accept/reject guard is unaffected.
        self.min_rel_delta: float = float(s2.get("merge_heal_min_rel_delta", 0.0))
        # LR schedule knobs (linear warmup → cosine decay → floor at lr_min).
        # All three are inert by default: warmup_steps=0 skips warmup,
        # decay_steps=0 skips cosine, lr_min defaults to `lr` so even a stray
        # decay term resolves to a constant lr. The three-knob inert state
        # reproduces the historical constant-LR Adam path bit-for-bit.
        # Types are coerced unconditionally (consistent with `lr` above);
        # a malformed YAML value crashes even with `enabled: false`.
        self.lr_warmup_steps: int = int(s2.get("merge_heal_lr_warmup_steps", 0))
        self.lr_min: float = float(s2.get("merge_heal_lr_min", self.lr))
        self.lr_decay_steps: int = int(s2.get("merge_heal_lr_decay_steps", 0))
        # Cross-domain telemetry knob (option A diagnostic). When enabled, the
        # heal capture phase also gathers a small WikiText (input, target) pool
        # per layer; `_heal_layer` evaluates a second `_holdout_mse_xd` against
        # it each eval — read-only, never gradients/accept-reject. Inert when
        # disabled. WikiText dataset/subset/split are inherited from the
        # thermometer config (top-level `stage6_validate.thermometer.wikitext`)
        # so heal and BPT eval use the same corpus identity.
        self.cross_domain_holdout_enabled: bool = bool(
            s2.get("merge_heal_cross_domain_holdout_enabled", False)
        )
        # Default 26214 = round(0.10 × default token_cap 262144) — matches the
        # Nemotron 10% holdout row count when both run at their defaults.
        self.xd_holdout_tokens: int = int(
            s2.get("merge_heal_xd_holdout_tokens", 26214)
        )
        # Shard storage knobs: heal activations live on disk as bf16 safetensors
        # shards (one safetensors file per `shard_rows` rows). The pool stops
        # scaling with RAM, so token_cap can be raised ~100× without OOM.
        self.shard_rows: int = int(s2.get("merge_heal_shard_rows", 4096))
        # ``None`` defers the path to the call site, which derives it from
        # ``artifacts_dir`` (canonical: ``artifacts_dir/_stage2_heal_shards``).
        # An explicit string overrides that — useful for routing shards to a
        # fast NVMe distinct from the artifacts NFS mount.
        _sd = s2.get("merge_heal_shard_dir")
        self.shard_dir: str | None = str(_sd) if _sd is not None else None
        # When False (default), each layer's shard dir is deleted after the
        # heal completes — bounded disk use. Flip True only for debugging.
        self.keep_shards: bool = bool(s2.get("merge_heal_keep_shards", False))

        if not self.enabled:
            return

        # Validate only when the feature is active so a disabled run with a
        # partially-specified block never fails.
        if self.grad_clip <= 0.0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_grad_clip={self.grad_clip}; "
                "must be > 0."
            )
        if self.lr <= 0.0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr={self.lr}; must be > 0."
            )
        for _idx, _beta in enumerate(self.adamw_betas):
            if not (0.0 <= _beta < 1.0):
                raise ValueError(
                    f"stage2_reap_ream.merge_heal_adamw_betas[{_idx}]={_beta}; "
                    "must be in [0.0, 1.0)."
                )
        if not (0.0 < self.holdout_fraction < 1.0):
            raise ValueError(
                f"stage2_reap_ream.merge_heal_holdout_fraction="
                f"{self.holdout_fraction}; must be in (0.0, 1.0)."
            )
        for _name, _val in (
            ("merge_heal_patience", self.patience),
            ("merge_heal_eval_interval", self.eval_interval),
            ("merge_heal_max_steps", self.max_steps),
            ("merge_heal_token_cap", self.token_cap),
            ("merge_heal_minibatch_size", self.minibatch_size),
            ("merge_heal_shard_rows", self.shard_rows),
        ):
            if _val < 1:
                raise ValueError(
                    f"stage2_reap_ream.{_name}={_val}; must be >= 1."
                )
        if not (0.0 <= self.min_rel_delta < 1.0):
            raise ValueError(
                f"stage2_reap_ream.merge_heal_min_rel_delta="
                f"{self.min_rel_delta}; must be in [0.0, 1.0)."
            )
        # The held-out eval (and hence accept/reject + save-best) only fires
        # at multiples of eval_interval. If eval_interval > max_steps it never
        # runs, so every layer would be silently REJECTED — fail fast instead.
        if self.eval_interval > self.max_steps:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_eval_interval={self.eval_interval} "
                f"exceeds merge_heal_max_steps={self.max_steps}; the held-out "
                "eval would never run and every layer would be rejected."
            )
        # LR-schedule validation. Layout is `[0..warmup) linear; [warmup,
        # warmup+decay] cosine; (warmup+decay..) flat at lr_min`.
        # The two count knobs are non-negative step counts; lr_min must be a
        # positive LR no larger than the peak `lr`; and an asymptote below the
        # peak (lr_min < lr) is unreachable without a cosine phase, so reject
        # the misconfig at construction.
        if self.lr_warmup_steps < 0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_warmup_steps="
                f"{self.lr_warmup_steps}; must be >= 0."
            )
        if self.lr_decay_steps < 0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_decay_steps="
                f"{self.lr_decay_steps}; must be >= 0."
            )
        if not (0.0 < self.lr_min <= self.lr):
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_min={self.lr_min}; "
                f"must satisfy 0 < lr_min <= merge_heal_lr ({self.lr})."
            )
        # --- Unreachable / no-op cosine guards (pair) ---
        if self.lr_min < self.lr and self.lr_decay_steps == 0:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_min={self.lr_min} < "
                f"merge_heal_lr={self.lr} but merge_heal_lr_decay_steps=0; "
                "the asymptote is unreachable. Set lr_decay_steps > 0 to "
                "schedule the descent, or set lr_min == merge_heal_lr."
            )
        if self.lr_warmup_steps >= self.max_steps:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_warmup_steps={self.lr_warmup_steps} "
                f">= merge_heal_max_steps={self.max_steps}; warmup would never "
                "complete before the hard step cap."
            )
        if self.lr_warmup_steps + self.lr_decay_steps > self.max_steps:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_warmup_steps + "
                f"merge_heal_lr_decay_steps ({self.lr_warmup_steps + self.lr_decay_steps}) "
                f"exceeds merge_heal_max_steps={self.max_steps}; the cosine schedule "
                "cannot reach lr_min within the hard step cap."
            )
        if self.lr_decay_steps > 0 and self.lr_min == self.lr:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_lr_decay_steps={self.lr_decay_steps} "
                f"but merge_heal_lr_min == merge_heal_lr ({self.lr}); the cosine "
                "schedule would be a no-op. Set lr_min < merge_heal_lr to enable "
                "decay, or set lr_decay_steps = 0."
            )
        if self.cross_domain_holdout_enabled and self.xd_holdout_tokens < 1:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_xd_holdout_tokens="
                f"{self.xd_holdout_tokens}; must be >= 1 when "
                "merge_heal_cross_domain_holdout_enabled is true."
            )
        # The whole-shard 90/10 split inside ShardWriter.finalize() needs at
        # least 2 shards' worth of captured rows. Catch the misconfig at
        # construction time so it fails before any corpus pass.
        if self.token_cap < 2 * self.shard_rows:
            raise ValueError(
                f"stage2_reap_ream.merge_heal_token_cap={self.token_cap} "
                f"must be >= 2 * merge_heal_shard_rows ({2 * self.shard_rows}); "
                "the whole-shard train/holdout split needs at least 2 shards."
            )
        if (
            self.cross_domain_holdout_enabled
            and self.xd_holdout_tokens < 2 * self.shard_rows
        ):
            raise ValueError(
                f"stage2_reap_ream.merge_heal_xd_holdout_tokens="
                f"{self.xd_holdout_tokens} must be >= 2 * merge_heal_shard_rows "
                f"({2 * self.shard_rows}) when "
                "merge_heal_cross_domain_holdout_enabled is true; the "
                "whole-shard train/holdout split needs at least 2 shards."
            )


def _make_shared_out_fn(layer_ref: MoELayerRef):
    """Build a closure that runs the frozen shared expert on a CPU fp32 tensor.

    The shared expert is protected by Stage 2 (untouched by REAP / REAM), so we
    feed activations through ``layer_ref.shared_expert`` (gated by
    ``shared_expert_gate`` when present) to reconstruct the shared contribution
    that the self-distillation target included. The closure handles device +
    dtype shuttling: input is moved to the expert's device + dtype for the
    matmul, output is cast back to fp32 on the input's original device.

    Returns ``None`` if the layer has no shared expert (the heal student just
    skips the shared addition in that case).
    """
    mlp = layer_ref.mlp
    shared_expert = layer_ref.shared_expert
    shared_gate = getattr(mlp, "shared_expert_gate", None)

    def _shared_out(x_in: torch.Tensor) -> torch.Tensor:
        if shared_expert is None:
            return torch.zeros_like(x_in)
        sp = next(shared_expert.parameters())
        sdev, sdtype = sp.device, sp.dtype
        # Heal runs in fp32, but the frozen shared expert keeps the model's
        # native dtype (bf16) — feed it in its own dtype to avoid a matmul
        # dtype mismatch, then cast back to fp32.
        with torch.no_grad():
            so = shared_expert(x_in.to(sdev, sdtype))
            if shared_gate is not None:
                so = torch.sigmoid(shared_gate(x_in.to(sdev, sdtype))) * so
        return so.to(x_in.device, torch.float32)

    return _shared_out


def _capture_mlp_io(
    model,
    layer_ref: MoELayerRef,
    calib_batches,
    *,
    device: torch.device,
    pool_size: int,
    shard_writer: ShardWriter,
) -> int:
    """Capture row-aligned ``(mlp_input, mlp_output)`` rows into ``shard_writer``.

    Runs a forward over ``calib_batches`` with :func:`early_exit_after_layer`
    so each pass stops right after ``layer_ref.layer_idx``. A forward hook on
    ``layer_ref.mlp`` records the MoE-block INPUT (``inputs[0]`` — the
    post-attention / post-layernorm hidden state fed into the block) and the
    MoE-block OUTPUT. This MUST be called BEFORE the layer is merged, while
    ``layer_ref.mlp`` is still its original pre-merge self — so the captured
    output is the self-distillation target.

    The hook appends each batch's flattened ``[N_tokens, hidden]`` rows to
    ``shard_writer``, which buffers and flushes safetensors shards on disk
    once ``shard_writer.shard_rows`` is reached. The forward loop stops as
    soon as ``shard_writer.n_captured >= pool_size``; a per-hook prefix slice
    truncates the final batch so the total never exceeds ``pool_size`` (exact
    cap, not approximate). The writer is left un-finalized — the caller is
    responsible for ``compute_shared_companions`` + ``finalize``. Returns the
    total rows captured.

    The captured prefix is an UNBIASED uniform sample of the calibration
    corpus because :func:`build_calibration_tensor` globally shuffles every
    calibration sequence with ``torch.randperm`` before they are batched.
    A contiguous prefix is therefore a uniform random sample; no reservoir
    sampling is needed (and adding it would be redundant).
    """
    layer_idx = layer_ref.layer_idx
    captured = 0

    def _hook(_module, inputs, output):
        nonlocal captured
        x_in = inputs[0]
        x_out = output[0] if isinstance(output, tuple) else output
        in_flat = x_in.reshape(-1, x_in.shape[-1]).detach()
        out_flat = x_out.reshape(-1, x_out.shape[-1]).detach()
        remaining = pool_size - captured
        if remaining <= 0:
            return
        if in_flat.size(0) > remaining:
            in_flat = in_flat[:remaining]
            out_flat = out_flat[:remaining]
        shard_writer.append(in_flat, out_flat)
        captured += int(in_flat.size(0))

    handle = layer_ref.mlp.register_forward_hook(_hook)
    was_training = model.training
    model.train(False)
    try:
        with early_exit_after_layer(model, layer_idx):
            for batch in calib_batches:
                if captured >= pool_size:
                    break
                with torch.no_grad():
                    try:
                        model(input_ids=batch.to(device))
                    except _EarlyExitException:
                        pass  # expected — target layer completed
    finally:
        handle.remove()
        if was_training:
            model.train()

    shard_writer.close_pending()

    if captured == 0:
        raise RuntimeError(
            f"merge-heal capture: layer {layer_idx} mlp forward hook never "
            "fired — calib_batches empty or layer never reached."
        )
    expected_hidden = layer_ref.router.weight.shape[-1]
    if shard_writer.hidden_dim != expected_hidden:
        raise RuntimeError(
            f"merge-heal capture: layer {layer_idx} writer hidden_dim "
            f"{shard_writer.hidden_dim} != model hidden size {expected_hidden} — "
            "shard_writer was constructed with the wrong hidden_dim."
        )
    if captured < pool_size:
        log.warning(
            "  merge-heal capture: layer %d — calibration data starved: "
            "%d token rows captured vs requested pool_size=%d; healing will "
            "proceed with the smaller pool.",
            layer_idx, captured, pool_size,
        )
    log.info(
        "  merge-heal capture: layer %d — %d (input,target) token rows "
        "captured to %d shards (pool_size=%d)",
        layer_idx, captured, len(shard_writer.shard_entries), pool_size,
    )
    return captured


def _heal_student_moe_output(
    *,
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
    esc_bias: torch.Tensor | None,
    expert_params: dict[int, dict[str, torch.Tensor]],
    centroid_order: list[int],
    top_k: int,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    """Faithful student replica of ``Qwen3_5MoeSparseMoeBlock.forward``.

    Reproduces the model's routed-expert path exactly (see
    ``Qwen3_5MoeTopKRouter.forward`` / ``Qwen3_5MoeSparseMoeBlock.forward``):
      1. ``router_logits = softmax(F.linear(x, router_weight) [+ bias], dim=-1)``
         — full softmax over all KEPT experts.
      2. ``topk_vals, topk_idx = topk(router_logits, top_k)``; then
         ``topk_vals /= topk_vals.sum(-1, keepdim=True)`` — the top-k
         renormalization the Qwen router applies.
      3. routed output ``= Σ_k topk_vals[:,k] · SwiGLU_{expert(k)}(x)``.
      4. block output ``= routed_output + shared_expert_output``.

    This returns that single block-output hidden-state tensor — the same
    quantity ``_capture_mlp_io`` records as the self-distillation target.

    The ``e_score_correction_bias`` term is added to the pre-softmax logits
    when the router exposes it (mirrors ``_router_routing_weights``); the base
    Qwen3.5 router has neither bias, so for that model both are ``None`` and
    this matches ``Qwen3_5MoeTopKRouter.forward`` bit-for-bit.

    Routed-expert dispatch is SPARSE: each expert's SwiGLU runs only on the
    token rows where that expert is in the top-k (``index_select`` to gather
    the subset, ``index_add`` to scatter the weighted result back). This
    matches the dense ``Σ_pos w·SwiGLU(x)`` formulation exactly — every token
    not selecting expert ``pos`` had weight 0 there — while avoiding the
    ``n_kept/top_k``× wasted SwiGLU compute of evaluating every expert on every
    token. Both ``index_select`` and ``index_add`` are differentiable, so
    gradients still flow to the trainable expert params.

    All inputs are fp32 and on the same device; returns fp32 ``[T, hidden]``.
    The shared-expert output is precomputed by the caller (it does not depend
    on the trainable params) and simply added back.
    """
    logits = F.linear(x, router_weight)
    if router_bias is not None:
        logits = logits + router_bias
    if esc_bias is not None:
        logits = logits + esc_bias
    router_probs = F.softmax(logits, dim=-1)  # (T, n_kept)
    k = min(top_k, router_probs.shape[-1])
    topk_vals, topk_idx = torch.topk(router_probs, k=k, dim=-1)  # (T, k)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    routed = torch.zeros_like(shared_out)
    n_kept = router_weight.shape[0]
    # Per-token per-position weight for expert `pos`: the renormalized top-k
    # value where this expert was selected, else 0. Summing over the k slots
    # collapses the (rare) case of the same expert appearing twice.
    for pos in range(n_kept):
        cid = centroid_order[pos]
        ep = expert_params[cid]
        sel = (topk_idx == pos)  # (T, k) bool
        w = (topk_vals * sel.to(topk_vals.dtype)).sum(dim=-1)  # (T,)
        rows = torch.nonzero(w, as_tuple=False).squeeze(-1)  # tokens routed here
        if rows.numel() == 0:
            continue
        x_sub = x.index_select(0, rows)
        e_out = _swiglu_forward(
            ep["gate_proj"], ep["up_proj"], ep["down_proj"], x_sub
        )
        contrib = w.index_select(0, rows).unsqueeze(-1) * e_out
        routed = routed.index_add(0, rows, contrib)
    return routed + shared_out


def _heal_lr_at_step(
    step: int,
    *,
    lr: float,
    lr_min: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Effective LR for merge-heal training at zero-indexed ``step``.

    Schedule layout (all step counts are zero-indexed and clamped):
      ``[0, warmup_steps)``                       linear  ``lr/warmup_steps → lr``
      ``[warmup_steps, warmup_steps+decay_steps)`` cosine  ``lr → lr_min``
      ``[warmup_steps+decay_steps, ∞)``            flat at ``lr_min``

    Linear-warmup convention: step ``s`` (0-indexed) takes
    ``lr * (s + 1) / warmup_steps`` — i.e. the first step is one step *into*
    the ramp (LR > 0), and step ``warmup_steps - 1`` lands exactly at ``lr``.
    Step ``warmup_steps`` is therefore the first cosine step (``t = 0`` →
    cosine value = ``lr``), giving continuity across the warmup/cosine
    boundary.

    Inert behaviours (all preserve the historical constant-LR Adam path):
      * ``warmup_steps == 0`` with ``decay_steps == 0`` or ``lr_min == lr``
        → constant ``lr``.
      * ``decay_steps == 0`` or ``lr_min == lr`` → constant ``lr`` after
        any warmup ramp.
    """
    if step < 0:
        raise ValueError(f"_heal_lr_at_step: step={step}; must be >= 0.")
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if decay_steps == 0:
        return lr
    t = step - warmup_steps
    if t >= decay_steps:
        return lr_min
    # Half-cosine from lr (t=0) to lr_min (t=decay_steps).
    cos_term = (1.0 + math.cos(math.pi * t / decay_steps)) * 0.5
    return lr_min + (lr - lr_min) * cos_term


def _heal_layer(
    *,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
    manifest: ShardManifest,
    manifest_dir: Path,
    heal_cfg: "_HealConfig",
    device: torch.device,
    xd_manifest: ShardManifest | None = None,
    xd_manifest_dir: Path | None = None,
) -> dict:
    """Per-layer merge-heal by SELF-DISTILLATION (disk-shard variant).

    Fine-tunes one already-merged layer so it reproduces its OWN pre-merge
    MoE-block output:
      * EVERY kept expert trains — merged centroids AND singletons. The router
        was resized, so even a singleton's contribution to the block output is
        perturbed and benefits from healing.
      * the resized ``router.weight`` (+ ``router.bias`` if present) trains
        only when ``heal_cfg.train_router`` is True; otherwise the router is
        frozen at its mechanical resize.

    Pool: ``manifest`` indexes per-layer ``(input, target, shared)`` shards on
    disk (bf16, written by :class:`ShardWriter`), already split 90/10 into
    train/holdout shards at finalize time. ``HealActivationDataset`` streams
    minibatches off disk per step and decodes to fp32 on-device — peak RAM is
    bounded by a small LRU shard cache, NOT by the total pool size.

    Loss = ``F.mse_loss(student_moe_out, target)``, where ``student_moe_out``
    is :func:`_heal_student_moe_output` over the streamed input rows + the
    precomputed shared-expert output that lives in the companion ``shared_*``
    shards. The shared addition was computed at capture time against the
    (Stage-2 protected) shared expert, so the heal target faithfully
    reproduces ``Qwen3_5MoeSparseMoeBlock.forward``.

    **Monotone-safe accept/reject:** the un-healed plain-merged weights are
    snapshotted before training; if the best healed held-out MSE does not
    beat the plain-merged held-out MSE the layer is REVERTED to the plain
    merge (worst case per layer == the no-heal baseline).

    ``xd_manifest`` is an optional cross-domain (e.g. WikiText) pool used for
    telemetry only — read-only MSE evaluation each eval interval, never
    backpropped and never consulted for accept/reject.

    Returns a JSON-able state dict:
        ``{"steps", "train_mse", "train_mse_at_best", "holdout_mse",
        "plain_merged_holdout_mse", "heal_gap", "accepted", "stop_reason",
        "holdout_mse_xd", "plain_merged_holdout_mse_xd"}``.
    """
    layer_idx = layer_ref.layer_idx
    router = layer_ref.router

    if manifest.layer_idx != layer_idx:
        raise RuntimeError(
            f"merge-heal layer {layer_idx}: manifest layer_idx "
            f"{manifest.layer_idx} != layer_ref.layer_idx — wrong manifest."
        )
    expected_hidden = layer_ref.router.weight.shape[-1]
    if manifest.hidden_dim != expected_hidden:
        raise RuntimeError(
            f"merge-heal layer {layer_idx}: manifest hidden_dim "
            f"{manifest.hidden_dim} != model hidden size {expected_hidden}."
        )

    banks = build_banks(layer_ref)
    # CRITICAL indexing contract: `_heal_layer` runs AFTER `bank.select(
    # final_kept_ids)` has rewritten the stacked expert tensors to hold only
    # the kept rows, re-indexed 0..n_kept-1. So banks here MUST be indexed by
    # post-select POSITION, not by the original expert id. Position `pos`
    # corresponds to original id `final_kept_ids[pos]`, and the resized router
    # row `pos` likewise — so `pos` is the consistent index across banks +
    # router. `expert_params` / `healed_experts` stay keyed by original
    # `cid` (that is what `centroid_order=final_kept_ids` expects downstream).
    pos_of: dict[int, int] = {cid: i for i, cid in enumerate(final_kept_ids)}
    bank_dtype = banks["gate_proj"].get(0).dtype

    # --- Trainable expert params: EVERY kept expert (merged + singleton) ----
    # expert_params holds a trainable fp32 leaf for every kept expert.
    expert_params: dict[int, dict[str, torch.Tensor]] = {}
    trainable_2d: list[nn.Parameter] = []
    healed_experts: list[int] = list(final_kept_ids)
    for pos, cid in enumerate(final_kept_ids):
        ep: dict[str, torch.Tensor] = {}
        for name in MATRIX_NAMES:
            # .detach() is essential: banks[name].get(pos) is a VIEW of the
            # model's stacked-experts nn.Parameter (requires_grad=True during
            # Stage 2). Without detach, loss.backward() would accumulate
            # gradients onto the full model's expert tensors — an unbounded
            # GPU memory leak across the heal run.
            w = banks[name].get(pos).detach().to(device, torch.float32).clone()
            p = nn.Parameter(w, requires_grad=True)
            trainable_2d.append(p)
            ep[name] = p
        expert_params[cid] = ep

    # --- Router: whole resized router.weight (+ bias) — trained iff flagged -
    trainable_1d: list[nn.Parameter] = []
    router_bias_p: torch.Tensor | None = None
    if heal_cfg.train_router:
        router_weight: torch.Tensor = nn.Parameter(
            router.weight.detach().to(device, torch.float32).clone(),
            requires_grad=True,
        )
        trainable_2d.append(router_weight)
        if getattr(router, "bias", None) is not None:
            router_bias_p = nn.Parameter(
                router.bias.detach().to(device, torch.float32).clone(),
                requires_grad=True,
            )
            trainable_1d.append(router_bias_p)
    else:
        # Frozen: detached tensors (no grad / not optimized) — still fed to
        # the student forward so routing is correct.
        router_weight = router.weight.detach().to(device, torch.float32)
        if getattr(router, "bias", None) is not None:
            router_bias_p = router.bias.detach().to(device, torch.float32)
    # e_score_correction_bias (if any) is part of the router's pre-softmax
    # path but is NOT trained here — kept frozen, mirrors _router_routing_weights.
    esc = getattr(router, "e_score_correction_bias", None)
    esc_bias = esc.detach().to(device, torch.float32) if esc is not None else None

    # Every kept expert is trainable, so trainable_2d is always non-empty —
    # the optimizer always has at least one parameter.
    top_k = layer_ref.top_k

    # --- Shard-backed activation dataset ----------------------------------
    # Pool rows live on disk as bf16 safetensors shards (one per ``shard_rows``
    # rows). ``HealActivationDataset`` streams minibatches per step into fp32
    # on-device tensors, with a small LRU shard cache so repeated reads of the
    # same shard within one step cost zero I/O. The 90/10 train/holdout split
    # was done at finalize time (whole-shard granularity, seeded by layer_idx),
    # so the heal step trains on ``manifest.train_shards`` and evaluates on
    # ``manifest.holdout_shards``.
    dataset = HealActivationDataset(
        manifest, manifest_dir, device=device, compute_dtype=torch.float32,
    )

    # Cross-domain (e.g. WikiText) dataset — telemetry only. Identical streaming
    # contract; evaluated independently of accept/reject and save-best.
    xd_dataset: HealActivationDataset | None = None
    if xd_manifest is not None:
        if xd_manifest_dir is None:
            raise ValueError(
                "_heal_layer: xd_manifest was provided but xd_manifest_dir is None — "
                "both are required to construct the cross-domain dataset."
            )
        if xd_manifest.hidden_dim != manifest.hidden_dim:
            raise ValueError(
                f"_heal_layer: cross-domain manifest hidden_dim "
                f"{xd_manifest.hidden_dim} != main manifest hidden_dim "
                f"{manifest.hidden_dim} — token alignment is broken."
            )
        xd_dataset = HealActivationDataset(
            xd_manifest, xd_manifest_dir, device=device, compute_dtype=torch.float32,
        )

    # Single fp32 AdamW over every trainable param (all kept experts, plus the
    # router when heal_cfg.train_router). weight_decay=0 — the self-distillation
    # target already anchors the weights to the pre-merge function.
    # Per-step LR follows `_heal_lr_at_step`; inert under defaults.
    opt = torch.optim.AdamW(
        trainable_2d + trainable_1d, lr=heal_cfg.lr,
        betas=heal_cfg.adamw_betas, weight_decay=0.0,
    )
    clip_params = trainable_2d + trainable_1d
    lr_schedule_active = (
        heal_cfg.lr_warmup_steps > 0
        or (heal_cfg.lr_decay_steps > 0 and heal_cfg.lr_min < heal_cfg.lr)
    )

    def _lr_at(s: int) -> float:
        return _heal_lr_at_step(
            s,
            lr=heal_cfg.lr,
            lr_min=heal_cfg.lr_min,
            warmup_steps=heal_cfg.lr_warmup_steps,
            decay_steps=heal_cfg.lr_decay_steps,
        )

    def _forward(x_in, shared_in):
        return _heal_student_moe_output(
            x=x_in,
            router_weight=router_weight,
            router_bias=router_bias_p,
            esc_bias=esc_bias,
            expert_params=expert_params,
            centroid_order=final_kept_ids,
            top_k=top_k,
            shared_out=shared_in,
        )

    # --- Minibatched held-out MSE ------------------------------------------
    # Evaluated in chunks of merge_heal_minibatch_size rows so the held-out
    # forward never materializes activations for the whole held-out set at
    # once — bounds peak memory on wide layers exactly like the train step.
    mb = max(1, heal_cfg.minibatch_size)

    @torch.no_grad()
    def _holdout_mse() -> float:
        sq_sum = 0.0
        n_elem = 0
        for xb, sb, tb in dataset.iter_holdout(batch_size=mb):
            pred = _forward(xb, sb)
            sq_sum += float(((pred - tb) ** 2).sum().item())
            n_elem += tb.numel()
        return sq_sum / max(1, n_elem)

    # Telemetry-only cross-domain holdout MSE; returns NaN when no xd dataset
    # was provided so the caller can treat the metric as missing.
    @torch.no_grad()
    def _holdout_mse_xd() -> float:
        if xd_dataset is None:
            return float("nan")
        sq_sum = 0.0
        n_elem = 0
        # Iterate the full XD pool (train + holdout shards); the train/holdout
        # split on this manifest is purely structural — telemetry has no
        # gradient-bearing distinction so all captured XD rows are eval rows.
        for xb, sb, tb in xd_dataset.iter_all(batch_size=mb):
            pred = _forward(xb, sb)
            sq_sum += float(((pred - tb) ** 2).sum().item())
            n_elem += tb.numel()
        return sq_sum / max(1, n_elem)

    # --- Best-snapshot bookkeeping -----------------------------------------
    # Snapshots are kept on CPU so the best-state copy does not double the
    # GPU footprint of the trainable params on the widest layer.
    def _snapshot() -> dict:
        return {
            "experts": {
                cid: {n: expert_params[cid][n].detach().to("cpu").clone()
                      for n in MATRIX_NAMES}
                for cid in healed_experts
            },
            # Router params are cloned only when trained — the restore block
            # below is guarded by `if heal_cfg.train_router`, so a frozen
            # router's snapshot data would be dead weight.
            "router_weight": (
                router_weight.detach().to("cpu").clone()
                if heal_cfg.train_router else None
            ),
            "router_bias": (
                router_bias_p.detach().to("cpu").clone()
                if (heal_cfg.train_router and router_bias_p is not None)
                else None
            ),
        }

    # plain_merged_* is the un-healed baseline for the monotone-safe accept/
    # reject guard: the initial holdout MSE + an independent weight snapshot.
    plain_merged_holdout_mse = _holdout_mse()
    plain_merged_state = _snapshot()
    best_holdout = plain_merged_holdout_mse
    best_state = _snapshot()
    # Cross-domain baseline + at-best snapshots, in the analogue role for the
    # WikiText pool. NaN whenever the cross-domain pool is not provided —
    # downstream consumers (logs + state dict) propagate the NaN to make
    # "metric absent" obvious.
    plain_merged_holdout_mse_xd = _holdout_mse_xd()
    xd_holdout_at_best = plain_merged_holdout_mse_xd
    # train_mse_at_best pairs with best_holdout — the train MSE recorded at the
    # SAME step the best held-out was seen, so heal_gap is a like-for-like
    # (held-out − train) comparison rather than best-vs-final.
    train_mse_at_best = float("nan")
    evals_since_improve = 0
    last_train_mse = float("nan")
    stop_reason = "max_steps"
    steps_done = 0

    # --- Minibatch sampler over the shard-backed training pool ------------
    # Each step pulls a fresh random minibatch of merge_heal_minibatch_size
    # rows from train shards; seeded for reproducibility across resumes.
    # ``sample_minibatch`` reads ``ceil(mb / shard_rows)`` random shards from
    # disk (with an LRU cache so a re-pick within a few steps is free) and
    # returns ``(xb, sb, tb)`` already on device in fp32. When the entire
    # training pool is smaller than ``mb`` it returns all rows.
    n_train = dataset.n_train
    mb_gen = torch.Generator(device="cpu").manual_seed(layer_idx + 1)

    # Most-recent applied LR; surfaced in the eval-time log line. Always
    # computed from `_lr_at(0)` so the value tracked in the log matches what
    # the optimiser will see on iter 0 (under inert defaults this equals
    # `heal_cfg.lr` exactly).
    last_lr = _lr_at(0)
    for step in range(heal_cfg.max_steps):
        if lr_schedule_active:
            last_lr = _lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = last_lr
        xb, sb, tb = dataset.sample_minibatch(mb, generator=mb_gen)
        opt.zero_grad(set_to_none=True)
        student = _forward(xb, sb)
        loss = F.mse_loss(student, tb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, heal_cfg.grad_clip)
        opt.step()
        steps_done = step + 1
        last_train_mse = float(loss.detach().item())

        if steps_done % heal_cfg.eval_interval == 0:
            h = _holdout_mse()
            # Snapshot the TRUE best on any strict improvement — the
            # accept/reject guard restores best_state and must not lag the
            # actual best. Patience reset, by contrast, requires a *meaningful*
            # improvement (>= min_rel_delta fraction of the running best);
            # this is the fix for the "noise-level gain resets patience
            # forever" bug. With min_rel_delta=0 the two checks coincide and
            # behaviour matches the historical strict-< criterion.
            strict_improved = h < best_holdout
            improved = h < best_holdout * (1.0 - heal_cfg.min_rel_delta)
            # Compute the cross-domain holdout once per eval (read-only); when
            # strict_improved fires, snapshot xd_holdout_at_best in lockstep
            # with best_holdout so the two metrics are sampled at the SAME step.
            h_xd = _holdout_mse_xd()
            if strict_improved:
                best_holdout = h
                best_state = _snapshot()
                train_mse_at_best = last_train_mse
                xd_holdout_at_best = h_xd
            if improved:
                evals_since_improve = 0
            else:
                evals_since_improve += 1
            # Per-eval progress line — without it a layer heals silently for up
            # to merge_heal_max_steps steps. Cadence == eval_interval. The
            # cross-domain `holdout_mse_xd` field is appended only when the
            # telemetry is active (it is NaN otherwise) — keeps the historical
            # log shape for inert runs.
            xd_log = (
                f" holdout_mse_xd={h_xd:.6e}" if not math.isnan(h_xd) else ""
            )
            log.info(
                "    heal layer %d: step %d/%d — train_mse=%.6e holdout_mse=%.6e"
                "%s best=%.6e %s (patience %d/%d, min_rel_delta=%.4f, lr=%.3e)",
                layer_idx, steps_done, heal_cfg.max_steps, last_train_mse, h,
                xd_log, best_holdout,
                "improved" if improved else "no-improve",
                evals_since_improve, heal_cfg.patience, heal_cfg.min_rel_delta,
                last_lr,
            )
            if not improved and evals_since_improve >= heal_cfg.patience:
                stop_reason = "patience"
                break

    # --- Accept/reject: monotone-safe guard --------------------------------
    # Accept the healed weights only if the best healed held-out MSE strictly
    # beats the plain-merged baseline; otherwise REVERT the layer to the plain
    # merge (worst case per layer == the no-heal baseline).
    accepted = bool(best_holdout < plain_merged_holdout_mse)
    restore_state = best_state if accepted else plain_merged_state

    # bank.set() casts dtype + moves device internally; the router params are
    # re-installed with an explicit dtype+device cast so they land back on the
    # router's original device (the snapshot lives on CPU).
    with torch.no_grad():
        for cid in healed_experts:
            pos = pos_of[cid]  # banks are post-select position-indexed
            for name in MATRIX_NAMES:
                banks[name].set(pos, restore_state["experts"][cid][name].to(bank_dtype))
        # Only reinstall the router when it was actually trained. With
        # train_router=False the router was never an optimizer parameter, so
        # router.weight/bias still hold exactly what the mechanical resize
        # left — reinstalling would be a wasteful no-op that also obscures the
        # frozen-router contract.
        if heal_cfg.train_router:
            _rw = router.weight
            router.weight = nn.Parameter(
                restore_state["router_weight"].to(device=_rw.device, dtype=_rw.dtype),
                requires_grad=_rw.requires_grad,
            )
            if getattr(router, "bias", None) is not None and restore_state["router_bias"] is not None:
                _rb = router.bias
                router.bias = nn.Parameter(
                    restore_state["router_bias"].to(device=_rb.device, dtype=_rb.dtype),
                    requires_grad=_rb.requires_grad,
                )

    # heal_gap pairs the best held-out MSE with the train MSE recorded at the
    # SAME step (train_mse_at_best). If no eval ever improved on the initial
    # snapshot (e.g. max_steps < eval_interval), there is no matching train
    # step — fall back to the final-step train MSE so the gap stays defined.
    gap_train_mse = (
        train_mse_at_best
        if not math.isnan(train_mse_at_best)
        else last_train_mse
    )
    heal_gap = best_holdout - gap_train_mse
    # End-of-layer summary: append the cross-domain pair only when present so
    # disabled runs keep the historical shape.
    xd_summary = (
        f" holdout_mse_xd={xd_holdout_at_best:.6e} "
        f"plain_merged_holdout_mse_xd={plain_merged_holdout_mse_xd:.6e}"
        if not math.isnan(plain_merged_holdout_mse_xd) else ""
    )
    log.info(
        "  merge-heal layer %d: %d steps (%s) — train_mse=%.6e (at_best=%.6e) "
        "holdout_mse=%.6e plain_merged_holdout_mse=%.6e heal_gap=%.6e%s "
        "accepted=%s (%d kept experts, train_router=%s)",
        layer_idx, steps_done, stop_reason, last_train_mse, gap_train_mse,
        best_holdout, plain_merged_holdout_mse, heal_gap, xd_summary, accepted,
        len(final_kept_ids), heal_cfg.train_router,
    )
    return {
        "steps": steps_done,
        "train_mse": last_train_mse,
        "train_mse_at_best": gap_train_mse,
        "holdout_mse": best_holdout,
        "plain_merged_holdout_mse": plain_merged_holdout_mse,
        "heal_gap": heal_gap,
        "accepted": accepted,
        "stop_reason": stop_reason,
        # NaN when cross-domain telemetry was disabled for this layer; live
        # consumers should treat NaN as "metric not collected".
        "holdout_mse_xd": xd_holdout_at_best,
        "plain_merged_holdout_mse_xd": plain_merged_holdout_mse_xd,
    }


def _summarize_distill_state(
    distill_state: dict[int, dict] | None,
) -> dict[str, int | float]:
    """Aggregate per-merged-group distillation outcomes into per-layer scalars
    for Trackio emission (spec § 5 step 7b / M8).

    Returns a dict with four keys:
        ``stage2/distill_groups``       — int, number of non-singleton groups
                                          actually distilled this layer.
        ``stage2/distill_mean_final_loss`` — float, mean of per-group ``final_loss``
                                          (NaN when no groups distilled).
        ``stage2/distill_mean_steps``   — float, mean step count across groups
                                          (reflects plateau-break behavior — ratio
                                          to ``expert_distill_steps`` shows how
                                          aggressively groups converged).
        ``stage2/distill_plateau_breaks`` — int, count of groups whose
                                          ``break_reason == "plateau"``.

    Returns an empty dict when ``distill_state is None`` (distillation
    disabled or no non-singleton groups). Caller's `_trackio_log({**existing, **summary})`
    pattern then naturally omits the keys for that layer.
    """
    if not distill_state:
        return {}
    groups = list(distill_state.values())
    # Skip "trivial" skips (singletons, zero-steps) so the means reflect actual
    # distillation work, not no-op placeholders.
    real = [g for g in groups if g.get("skip") != "trivial" and g.get("final_loss") is not None]
    if not real:
        return {
            "stage2/distill_groups": 0,
            "stage2/distill_mean_final_loss": float("nan"),
            "stage2/distill_mean_steps": 0.0,
            "stage2/distill_plateau_breaks": 0,
        }
    n = len(real)
    final_losses = [float(g["final_loss"]) for g in real]
    steps = [int(g.get("steps", 0)) for g in real]
    plateaus = sum(1 for g in real if g.get("break_reason") == "plateau")
    return {
        "stage2/distill_groups": n,
        "stage2/distill_mean_final_loss": sum(final_losses) / n,
        "stage2/distill_mean_steps": sum(steps) / n,
        "stage2/distill_plateau_breaks": plateaus,
    }


class MergeHealPlugin(Stage2Plugin):
    """Per-layer merge-heal (Task 17 of the plugin-architecture refactor).

    T17 status: scaffold only — NOT on the live phase walk.
    ``LegacyAdapter.pre_merge_snapshot`` still calls ``_capture_mlp_io``,
    ``LegacyAdapter.post_merge`` still calls ``_make_shared_out_fn`` +
    ``_heal_layer``, and ``LegacyAdapter.write_artifacts`` still calls
    ``_summarize_distill_state`` — all via a late
    ``from ...stage2_reap_ream import ...``. The ``MOE_STAGE2_LEGACY_LOOP=1``
    path in ``stage2_reap_ream.run()`` calls all five directly too. This class
    exists so T18 has a per-layer merge-heal plugin to wire into the
    decomposed ``pre_merge_snapshot`` / ``post_merge`` / ``write_artifacts``
    phases.

    Config gate: enabled iff ``stage2_reap_ream.merge_heal_enabled`` is truthy.
    Unlike ``ExpertDistillPlugin`` / ``EmRefinePlugin`` (numeric-threshold
    knobs), ``merge_heal_enabled`` is a plain boolean flag (default False), so
    the base ``Stage2Plugin.is_enabled`` (AND-of-truthy-``enabled_by``-flags)
    expresses the gate exactly — no ``is_enabled`` override is needed. This is
    the first plugin in the refactor to use a non-empty ``enabled_by`` tuple
    with the base ``is_enabled``.
    """

    name = "merge_heal"
    # Plain bool flag: the base AND-of-flags is_enabled works directly.
    enabled_by: tuple[str, ...] = ("merge_heal_enabled",)

    def pre_merge_snapshot(self, ctx: PipelineContext) -> None:
        """Documented no-op for T17.

        The live pre-merge mlp-I/O capture still belongs to
        ``LegacyAdapter.pre_merge_snapshot`` (and the
        ``MOE_STAGE2_LEGACY_LOOP=1`` path), which call ``_capture_mlp_io``
        directly. Returning ``None`` makes this hook a clean no-op. T18 wires
        the real call here once ``pre_merge_snapshot`` is decomposed.
        """
        return None

    def post_merge(self, ctx: PipelineContext) -> None:
        """Documented no-op for T17.

        The live per-layer heal still belongs to ``LegacyAdapter.post_merge``
        (and the ``MOE_STAGE2_LEGACY_LOOP=1`` path), which call
        ``_make_shared_out_fn`` + ``_heal_layer`` directly. Returning ``None``
        makes this hook a clean no-op. T18 wires the real call here.
        """
        return None

    def write_artifacts(self, ctx: PipelineContext, partial_dir: Any) -> dict[str, Any]:
        """Documented no-op for T17.

        The live ``_summarize_distill_state`` telemetry emission still belongs
        to ``LegacyAdapter.write_artifacts``. T17 returns an empty dict (the
        base contract's neutral value) so the plugin contributes nothing to
        the merged artifact payload until T18 decomposes this phase.
        """
        return {}

