"""Phase A — MA-formation layer detection (dual-signal OR rule).

Paper: ALGORITHM_REFERENCE.md §4 Phase A + D-ma-detector. Migrated from
the legacy Stage 1 module (the Phase-A block) in sub-task 9 of the Stage 1 →
plugin-architecture refactor — the LAST phase migration.

The plugin's externally observable behaviour is byte-identical to the
legacy inline Phase A: same two forward hooks (decoder-layer output +
MoE-block output), same early-exit calibration pass, same dual-signal OR
rule, same first-layer Q99 absolute-outlier check, same 0.75-depth
fallback when L is empty. Verified via the golden snapshot test (the
``dual_signal`` block of ``stage1_blacklist.json``).

Phase A runs its OWN dedicated calibration pass — ``provides`` is
empty. It cannot use the shared ``CalibrationEngine`` because it hooks
whole decoder-layer / MoE-block module outputs, not the per-expert
channels the engine wires (see subtask_9_plan.md §2.2).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from ...pipeline.safe_json import safe_float
from ...utils.activation_hooks import run_calibration_early_exit
from ...utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from ...utils.model_io import iter_decoder_layers, iter_moe_layers
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)

# --- spec-pinned thresholds (moved verbatim from the legacy Stage 1 module) ---
_MA_RATIO = 100.0                     # max / Q99 threshold — first MoE layer absolute outlier check
_MA_GROWTH_RATIO = 3.0                # was 5.0; calibrated for Qwen3.5/3.6 attn_output_gate=true
_MOE_OUTPUT_GROWTH_RATIO = 2.0        # ungated MoE-block-output secondary signal (OR with residual)
_PHASE_A_BATCH_SIZE = 32              # Phase A has zero accumulators on GPU besides max-magnitude


def _phase_a_progress_cb(n_total: int, log_every: int = 64):
    """Phase-A-specialised Trackio progress callback (see subtask_9_plan.md §2.5).

    Byte-identical Trackio semantics to the legacy ``_make_calibration_progress_cb``
    with ``phase_tag="phase_a"``: emits ``stage1/phase_a/calibration_progress``
    (fraction) + ``stage1/phase_a/calibration_step`` (raw index) every
    ``log_every`` batches, non-blocking (no flush).
    """
    def _cb(i: int) -> None:
        n_done = i + 1
        if log_every > 0 and n_done % log_every == 0:
            _trackio_log({
                "stage1/phase_a/calibration_progress": n_done / n_total,
                "stage1/phase_a/calibration_step": n_done,
            })
    return _cb


class MADetectionPlugin:
    """MA-formation layer detector (Phase A — dual-signal OR rule).

    Mandatory — :meth:`is_enabled` returns ``True`` unconditionally
    (Phase A always runs; there is no flag). Runs its own dedicated
    early-exit calibration pass; ``provides`` is empty.

    Reads ``model`` / ``tokenizer`` / ``config`` / ``artifacts_dir`` /
    ``device`` from the context; writes the four Phase-A outputs:
    ``L`` (set[int]), ``residual_growth`` / ``moe_output_growth`` /
    ``moe_output_max`` (dict[int, float]).

    Contributes the ``dual_signal`` block of ``stage1_blacklist.json``
    via :meth:`contribute_artifact` (3 keys, NaN/Inf → JSON null).

    The declared ``reads`` tuple lists ``config`` (not a pre-built
    ``calibration_spec``) — the plugin builds the spec + tensor + batches
    internally, matching ``AblationFilterPlugin``. Sub-task 10's
    orchestrator passes ``model`` / ``tokenizer`` / ``config`` on the ctx
    exactly as the legacy delegation block does.
    """

    name: str = "ma_detection"
    paper: str = "MA-formation dual-signal detector (ALGORITHM_REFERENCE.md §4 Phase A, D-ma-detector)"
    config_key: str = "stage1_grape.super_expert_detection"
    reads: tuple[str, ...] = ("model", "tokenizer", "config", "artifacts_dir", "device")
    writes: tuple[str, ...] = ("L", "residual_growth", "moe_output_growth", "moe_output_max")
    # Phase A runs its OWN dedicated early-exit forward pass — it does not
    # consume any shared accumulator from Phase B's CalibrationEngine. See
    # subtask_9_plan.md §2.2 for why the hook semantics differ.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Mandatory — Phase A always runs. There is no flag."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Execute Phase A end-to-end: build calibration batches, run the
        dedicated early-exit pass with decoder + MoE hooks, detect L.

        Reads ``model`` / ``tokenizer`` / ``config`` / ``artifacts_dir`` /
        ``device``; writes ``L`` / ``residual_growth`` /
        ``moe_output_growth`` / ``moe_output_max``.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config: dict = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        device = ctx.get("device")

        s1 = config["stage1_grape"]
        cal = config["calibration"]
        se_cfg = s1.get("super_expert_detection", {})

        moe_layers = list(iter_moe_layers(model))
        if not moe_layers:
            raise ValueError(
                "Stage 1 Phase A: model has no MoE layers — check iter_moe_layers() "
                "compatibility with this model architecture."
            )

        ma_ratio = float(se_cfg.get("ma_ratio", _MA_RATIO))
        ma_growth_ratio = float(se_cfg.get("ma_growth_ratio", _MA_GROWTH_RATIO))
        moe_output_growth_ratio = float(
            se_cfg.get("moe_output_growth_ratio", _MOE_OUTPUT_GROWTH_RATIO)
        )

        # Build the calibration tensor + Phase-A batches (verbatim from the
        # legacy Stage 1 run() spec-build). seed_offset=1 is spec-pinned.
        spec = spec_from_config(
            cal,
            num_sequences_override=s1.get("num_calibration_samples"),
            seed_offset=1,
        )
        calib = build_calibration_tensor(
            tokenizer, spec,
            cache_dir=artifacts_dir / "_calibration_cache",
        )
        phase_a_batch_size = int(s1.get("phase_a_batch_size", _PHASE_A_BATCH_SIZE))
        batches = iter_batches(calib, batch_size=phase_a_batch_size)

        log.info(
            "Stage 1 Phase A: detecting MA-formation layers over %d samples (%d MoE layers)",
            len(batches), len(moe_layers),
        )

        L, residual_growth, moe_output_growth, moe_output_max = _detect_ma_layers(
            model, batches, moe_layers, device,
            ma_ratio=ma_ratio,
            ma_growth_ratio=ma_growth_ratio,
            moe_output_growth_ratio=moe_output_growth_ratio,
        )
        log.info("Stage 1 Phase A: MA-formation layers L = %s", sorted(L))

        ctx.set("L", L)
        ctx.set("residual_growth", residual_growth)
        ctx.set("moe_output_growth", moe_output_growth)
        ctx.set("moe_output_max", moe_output_max)

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the ``dual_signal`` block of ``stage1_blacklist.json``.

        Three keys; per-layer growth ratios, NaN/±Inf serialised as JSON
        null (the first MoE layer's ``residual_growth`` entry is NaN by
        design — ``_detect_ma_layers`` has no predecessor for it).
        """
        residual_growth: dict[int, float] = ctx.get("residual_growth")
        moe_output_growth: dict[int, float] = ctx.get("moe_output_growth")
        moe_output_max: dict[int, float] = ctx.get("moe_output_max")
        return {
            "residual_growth_per_layer": {
                str(li): safe_float(v) for li, v in residual_growth.items()
            },
            "moe_output_growth_per_layer": {
                str(li): safe_float(v) for li, v in moe_output_growth.items()
            },
            "moe_output_max_per_layer": {
                str(li): safe_float(v) for li, v in moe_output_max.items()
            },
        }


# ---------------------------------------------------------------------------
# Phase A: MA-formation layer detection — module-level private helpers.
# Moved verbatim from the legacy Stage 1 module (sub-task 9). This module is
# the single source of truth.
# ---------------------------------------------------------------------------


def _flag_layer_dual_signal(
    *,
    residual_ratio: float,
    moe_ratio: float,
    residual_threshold: float,
    moe_threshold: float,
) -> bool:
    """OR-rule flagging: True iff EITHER signal exceeds its threshold (D-ma-detector)."""
    return (residual_ratio > residual_threshold) or (moe_ratio > moe_threshold)


def _detect_ma_layers(
    model: nn.Module,
    batches,
    moe_layers,
    device,
    *,
    ma_ratio: float = _MA_RATIO,
    ma_growth_ratio: float = _MA_GROWTH_RATIO,
    moe_output_growth_ratio: float = _MOE_OUTPUT_GROWTH_RATIO,
) -> tuple[set[int], dict[int, float], dict[int, float], dict[int, float]]:
    """Forward pass 1: identify MA-formation layers via dual-signal OR rule.

    Returns (L, residual_growth, moe_output_growth, moe_output_max):
      L                   — set of MA-formation layer indices.
      residual_growth     — per-MoE-layer max|H_l|/max|H_{l-1}| (residual stream).
                            First MoE layer entry is float('nan') (no predecessor).
      moe_output_growth   — per-MoE-layer max|MoE_l|/max|MoE_{l-1}| (post-routing-weighted-sum,
                            pre-residual-add). First MoE layer entry is 0.0 (no predecessor).
      moe_output_max      — per-MoE-layer raw max|MoE_l| (for diagnostics).

    See ALGORITHM_REFERENCE.md §4 Phase A and D-ma-detector for the OR rule rationale.
    """
    sorted_moe_layer_indices = sorted(ref.layer_idx for ref in moe_layers)
    if not sorted_moe_layer_indices:
        return set(), {}, {}, {}
    first_moe_layer_idx = sorted_moe_layer_indices[0]
    moe_layer_by_idx = {ref.layer_idx: ref for ref in moe_layers}

    decoder_layers: list[tuple[int, nn.Module]] = list(iter_decoder_layers(model))
    if not decoder_layers:
        raise ValueError("_detect_ma_layers: no decoder layers found")
    decoder_layer_modules = {layer: idx for idx, layer in decoder_layers}
    if len(decoder_layer_modules) != len(decoder_layers):
        raise ValueError("_detect_ma_layers: weight-tied decoder layers detected")
    sorted_decoder_layer_indices = sorted(decoder_layer_modules.values())

    # Residual-stream max: hooked on every decoder layer
    layer_max: dict[int, float] = {idx: 0.0 for idx in sorted_decoder_layer_indices}
    # MoE-block-output max: hooked on each MoE layer's `mlp` (Qwen3_5MoeSparseMoeBlock)
    moe_block_max: dict[int, float] = {idx: 0.0 for idx in sorted_moe_layer_indices}
    first_layer_q99_buffer: list[np.ndarray] = []
    handles: list = []

    def _make_decoder_hook(layer_idx: int):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                return
            h_abs = h.detach().abs().float()
            curr_max = h_abs.max().item()
            if curr_max > layer_max[layer_idx]:
                layer_max[layer_idx] = curr_max
            if layer_idx == first_moe_layer_idx:
                first_layer_q99_buffer.append(h_abs.flatten().cpu().numpy())
        return _hook

    def _make_moe_hook(layer_idx: int):
        def _hook(_module, _input, output):
            # Qwen3_5MoeSparseMoeBlock.forward returns (hidden_states, router_logits) tuple.
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                return
            curr_max = h.detach().abs().float().max().item()
            if curr_max > moe_block_max[layer_idx]:
                moe_block_max[layer_idx] = curr_max
        return _hook

    for module, layer_idx in decoder_layer_modules.items():
        handles.append(module.register_forward_hook(_make_decoder_hook(layer_idx)))
    for layer_idx in sorted_moe_layer_indices:
        ref = moe_layer_by_idx[layer_idx]
        handles.append(ref.mlp.register_forward_hook(_make_moe_hook(layer_idx)))

    try:
        # Phase A only needs decoder-layer and MoE-block activations — lm_head
        # is not needed and on 80 GB GPUs its logits tensor (~30 GB for
        # batch×seq×vocab) causes OOM. Stop after the last decoder layer.
        run_calibration_early_exit(
            model, batches,
            target_layer_idx=sorted_decoder_layer_indices[-1],
            device=device,
            per_batch_callback=_phase_a_progress_cb(n_total=len(batches)),
        )
    finally:
        for h in handles:
            h.remove()

    if first_layer_q99_buffer:
        first_layer_q99 = float(
            np.percentile(np.concatenate(first_layer_q99_buffer), 99.0)
        )
    else:
        first_layer_q99 = 0.0

    L: set[int] = set()
    residual_growth: dict[int, float] = {}
    moe_output_growth: dict[int, float] = {}
    decoder_index_pos = {idx: pos for pos, idx in enumerate(sorted_decoder_layer_indices)}

    for layer_idx in sorted_moe_layer_indices:
        if layer_idx == first_moe_layer_idx:
            residual_growth[layer_idx] = float("nan")
            moe_output_growth[layer_idx] = 0.0
            if first_layer_q99 <= 0:
                log.warning(
                    "Stage 1: first-MoE-layer Q99 is %.2e for layer %d; model output may be "
                    "degenerate — excluding from MA-formation candidate set L",
                    first_layer_q99, layer_idx,
                )
            elif layer_max[layer_idx] > ma_ratio * first_layer_q99:
                L.add(layer_idx)
            continue
        # Residual ratio against the immediately preceding decoder layer
        pos = decoder_index_pos[layer_idx]
        prev_decoder_idx = sorted_decoder_layer_indices[pos - 1]
        prev_max = layer_max[prev_decoder_idx]
        res_ratio = (layer_max[layer_idx] / prev_max) if prev_max > 0 else 0.0
        residual_growth[layer_idx] = res_ratio
        # MoE ratio against the previous MoE layer in the sorted order
        prev_moe_pos = sorted_moe_layer_indices.index(layer_idx) - 1
        prev_moe_idx = sorted_moe_layer_indices[prev_moe_pos]
        prev_moe_max = moe_block_max[prev_moe_idx]
        moe_ratio = (moe_block_max[layer_idx] / prev_moe_max) if prev_moe_max > 0 else 0.0
        moe_output_growth[layer_idx] = moe_ratio
        if _flag_layer_dual_signal(
            residual_ratio=res_ratio,
            moe_ratio=moe_ratio,
            residual_threshold=ma_growth_ratio,
            moe_threshold=moe_output_growth_ratio,
        ):
            L.add(layer_idx)

    if not L:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
        total_layers = getattr(text_cfg, "num_hidden_layers", None) or len(sorted_decoder_layer_indices)
        cutoff = round(0.75 * float(total_layers))
        fallback_layers = [li for li in sorted_moe_layer_indices if li < cutoff]
        log.warning(
            "Stage 1 Phase A: dual-signal detector returned ∅; falling back to "
            "layer_idx < round(0.75 × %d) = %d. Fallback L = %s",
            total_layers, cutoff, fallback_layers,
        )
        L = set(fallback_layers)

    return L, residual_growth, moe_output_growth, dict(moe_block_max)
