"""MA-formation layer detection (dual-signal OR rule).

Paper
-----
Su et al., "Super Experts in MoE Models", arXiv:2507.23279 (2025).
Algorithm 1 (Appendix L) **Stage 1** of the paper — the calibration loop
that constructs the set ``L`` of MA-formation layers:

    L ← ∅
    for each batch x ∈ D:
        for each layer l in the model:
            Compute hidden activations H_l(x)
            if MA pattern detected in H_l(x):
                L ← L ∪ {l}
            end if
        end for
    end for

(verbatim from ``audit/spec_compliance/01_papers/2507.23279/source.md``
lines 1962-1976.) Super experts (SEs) are defined only on layers in
``L`` — the paper's Algorithm 1 Stage 2 block (source.md lines 1978-2003)
loops ``for each layer l ∈ L``. The paper documents that ``L`` is
empirically tiny — Mixtral-8x7B-Instruct has its single SE at L1E3
(Table 2, source.md line 540); Qwen3-30B-A3B at L1E68, L2E92, L3E82
(same table) — because MA-formation begins in the first 1-3 decoder
layers then propagates stably via residuals. The paper's §3.2.1
(source.md line 386) and §3.2.2 (line 410) elaborate; the "input-stable
across datasets" claim is at source.md line 405-406.

Official implementation (golden reference)
------------------------------------------
``github.com/ZunhaiSu/Super-Experts-Profilling`` pinned to commit
``573aead3127ae593ba267758b832944f8fed1485`` (default branch ``main``
HEAD, dated 2025-09-25). Two artefacts matter for this plugin:

* ``run.py:28`` declares the CLI flag
  ``--include_layers type=float default=0.75``.
* ``eval_utils.py:470-471`` defines
  ``_super_experts_analysis(..., include_layers=0.75)`` where the first
  body line is ``include_layers = round(total_layers * include_layers)``,
  and line 479 filters layers via ``if int(each['layer_index']) <
  include_layers:``.

The official code therefore **does not implement dynamic MA-formation
detection at all** — it uses a fixed depth heuristic
``layer_index < round(0.75 × total_layers)`` to populate ``L``.

This plugin's deviation from paper + official code (D-ma-detector)
-------------------------------------------------------------------
**Paper says:** Algorithm 1 line 8 is the deliberately-undefined
``if MA pattern detected`` — no formula. The paper PDF and source.md
contain no numeric thresholds for MA detection.

**Official code does:** a fixed depth heuristic ``layer_index <
round(0.75 × total_layers)`` — no dynamic per-batch detection.

**This implementation does:** a dynamic dual-signal detector. A layer
``l`` is added to ``L`` if it is actively FORMING (amplifying) a
massive activation — not merely propagating one that formed earlier.
The detector is primary; the 0.75-depth heuristic is the secondary
fallback when the dynamic detector returns ∅.

* **First MoE layer** (no predecessor to compare): absolute-outlier
  check — ``max|H_l(x)| > _MA_RATIO × Q_99(|H_l(x)|)``. The LHS is the
  max magnitude across all calibration tokens; the RHS is the 99th
  percentile across all token magnitudes seen during the pass.
  ``_MA_RATIO = 100`` is project-chosen.
* **All subsequent layers** — dual-signal OR rule, add ``l`` to ``L``
  if EITHER condition holds:
    - Residual-stream growth (primary, gated):
      ``max|H_l(x)| / max|H_{l-1}(x)| > _MA_GROWTH_RATIO``.
      ``_MA_GROWTH_RATIO = 3.0`` is **project-chosen for
      attn_output_gate=true architectures** (Qwen3.5/3.6); motivated
      by the gated residual's ≈0.6× sigmoid-gate attenuation
      (``E[sigmoid(gate)] ≈ 0.6 × 5.0 ≈ 3.0``).

      **Project history of this threshold** (paper Algorithm 1 line 8
      gives no number — both 5.0 and 3.0 are project-original
      empirical calibrations):
        * Commit ``3db7d80`` ("fix(stage1): Phase A detection —
          growth-based MA formation vs propagation") **introduced**
          ``_MA_GROWTH_RATIO = 5.0`` with the inline source comment
          "implementation choice" — first per-batch growth-based
          detector, calibrated for Qwen3-30B-A3B (un-gated).
        * Commit ``40956e3`` ("diag: standalone Phase A MA-formation
          detector + HF Jobs entrypoint") kept 5.0 as the production
          value on Qwen3-30B with the inline note "tuned on Qwen3-30B
          without [gated attention]".
        * Commit ``172e72e`` ("feat(stage1): dual-signal Phase A
          (residual OR MoE-output)") **recalibrated to 3.0** for the
          target architectures (Qwen3.5/3.6 with attn_output_gate=
          true) and added the MoE-output secondary signal below.

      Operators porting to un-gated architectures (LLaMA, Mixtral,
      Qwen3-30B) should reset ``_MA_GROWTH_RATIO`` to 5.0 (the prior
      project-calibrated un-gated value) and re-validate; the 3.0
      value is correct only under sigmoid-gate-attenuated residuals.
    - MoE-output growth (secondary, ungated):
      ``max|MoE_l(x)| / max|MoE_{l-1}(x)| > _MOE_OUTPUT_GROWTH_RATIO``.
      ``_MOE_OUTPUT_GROWTH_RATIO = 2.0`` is project-chosen — the
      ungated MoE-branch signal has a smaller dynamic range than the
      gated residual stream, so the threshold is set lower; tuned
      empirically to maximise recall under the OR rule.
* **Fallback** (only when the dynamic detector returns ∅): the
  official-code depth heuristic — keep layers with
  ``index < round(0.75 × total_layers)``. This guarantees ``L`` is
  never empty.

**Architecture context (Qwen3.5/3.6 specific):** full-attention layers
(1 of every 4) apply ``attn_output_gate=true`` (sigmoid gate on attn
output before residual add); the other 3-of-4 linear-attention layers
apply ``FusedRMSNormGated`` (combined RMS-norm + sigmoid gate). At
linear-attention layers the attention contribution is magnitude-flat,
so the residual-stream growth signal is driven exclusively by the MoE
branch — which is exactly why the MoE-output secondary signal exists.
The full-attention "reset" layer (e.g. L31 in Qwen3.6) can show an
aggressive residual-stream **collapse**; the detector correctly
ignores collapses (it flags growth, not drops).

Sampling parameters (project-chosen — paper does not prescribe)
---------------------------------------------------------------
* ``_PHASE_A_BATCH_SIZE = 32`` — only max magnitudes are tracked, so
  batch size is invariant within VRAM headroom.
* ``num_calibration_samples`` from config (production: 1024) —
  saturates the per-layer max accumulator at low cost; the AIMER paper
  (arXiv:2603.18492) calibration-sensitivity figure reports < 5%
  Frobenius drift between 1024 and 4000 samples.

Output context slots
--------------------
Writes to the ``PipelineContext``:
  * ``L`` — ``set[int]``, the MA-formation layer set.
  * ``residual_growth`` — ``dict[int, float]``, per-layer residual
    growth ratio (NaN for the first MoE layer; no predecessor).
  * ``moe_output_growth`` — ``dict[int, float]``, per-layer MoE-output
    growth ratio (0.0 for the first MoE layer).
  * ``moe_output_max`` — ``dict[int, float]``, per-layer MoE-block
    maximum magnitude (diagnostic).

Artifact contribution: the ``dual_signal`` block of
``stage1_blacklist.json`` (3 keys: ``residual_growth``,
``moe_output_growth``, ``moe_output_max``; NaN/Inf → JSON null).

Calibration pass discipline
---------------------------
Runs its OWN dedicated early-exit forward pass — ``provides`` is
empty. It cannot use the shared ``CalibrationEngine`` because it
hooks whole decoder-layer / MoE-block module outputs, not the per-
expert channels the engine wires.

Naming-history note
-------------------
Code-level identifiers carrying the ``phase_a`` prefix
(``_PHASE_A_BATCH_SIZE``, ``_phase_a_progress_cb``, Trackio keys
``stage1/phase_a/calibration_progress`` and
``stage1/phase_a/calibration_step``) are kept as-is. They date from
when this concern was "Phase A" of the pre-refactor Stage 1 monolith;
renaming them now would invalidate operator Trackio dashboards that
key on those metric names. Treat them as opaque historical tags —
the implementation concern is "MA-formation layer detection", not
"Phase A".
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

# --- detector thresholds (paper has no number; see module docstring + git
# history of this file for the project-original derivation of each value) ---
_MA_RATIO = 100.0                     # max / Q99 threshold — first MoE layer absolute outlier check
_MA_GROWTH_RATIO = 3.0                # was 5.0; recalibrated for Qwen3.5/3.6 attn_output_gate=true in commit 172e72e
_MOE_OUTPUT_GROWTH_RATIO = 2.0        # ungated MoE-block-output secondary signal (OR with residual)
_PHASE_A_BATCH_SIZE = 32              # name preserved for Trackio key stability — see naming-history note in module docstring


def _phase_a_progress_cb(n_total: int, log_every: int = 64):
    """Trackio progress callback for the dedicated MA-formation forward pass.

    Emits ``stage1/phase_a/calibration_progress`` (fraction in [0, 1]) and
    ``stage1/phase_a/calibration_step`` (raw index) every ``log_every``
    batches, non-blocking (no flush). The ``phase_a`` Trackio key is
    historical (Trackio dashboards depend on it); the function/constant
    names retain the ``_phase_a`` prefix for the same reason — see the
    module docstring's *Naming-history note* below.
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
    """MA-formation layer detector (dual-signal OR rule).

    See this module's docstring for paper + official-code + deviation
    rationale (the implementation is project-original; the paper
    deliberately leaves Algorithm 1 line 8 — ``if MA pattern
    detected`` — undefined, and the official code uses only a fixed
    depth heuristic with no dynamic detection).

    Mandatory plugin — :meth:`is_enabled` returns ``True``
    unconditionally; MA-formation detection always runs because every
    downstream SE-detection plugin reads ``L`` from the context.
    Runs its own dedicated early-exit calibration pass; ``provides``
    is empty.

    Reads ``model`` / ``tokenizer`` / ``config`` / ``artifacts_dir`` /
    ``device`` from the context. Writes the four output slots: ``L``
    (set[int]), ``residual_growth`` / ``moe_output_growth`` /
    ``moe_output_max`` (each ``dict[int, float]``).

    Contributes the ``dual_signal`` block of ``stage1_blacklist.json``
    via :meth:`contribute_artifact` (3 keys; NaN/±Inf → JSON null).

    The declared ``reads`` tuple lists ``config`` (not a pre-built
    ``calibration_spec``) — the plugin builds the spec + tensor +
    batches internally, matching ``AblationFilterPlugin``.
    """

    name: str = "ma_detection"
    paper: str = (
        "Su et al., 'Super Experts in MoE Models' (arXiv:2507.23279, 2025), "
        "Algorithm 1 (Appendix L) Stage 1 — MA-formation layer detection. "
        "Official code: github.com/ZunhaiSu/Super-Experts-Profilling "
        "@ commit 573aead3127ae593ba267758b832944f8fed1485 (2025-09-25) — "
        "implements only a fixed 0.75-depth heuristic, not dynamic "
        "detection. This plugin's dual-signal OR detector + 0.75-depth "
        "fallback is project-original (paper Algorithm 1 line 8 is "
        "deliberately undefined). See this module's docstring for "
        "paper-cited line numbers (verified against "
        "audit/spec_compliance/01_papers/2507.23279/source.md), the "
        "official-code citations, and the git-archaeology history of "
        "the threshold values."
    )
    config_key: str = "stage1_grape.super_expert_detection"
    reads: tuple[str, ...] = ("model", "tokenizer", "config", "artifacts_dir", "device")
    writes: tuple[str, ...] = ("L", "residual_growth", "moe_output_growth", "moe_output_max")
    # This plugin runs its OWN dedicated early-exit forward pass — it does
    # not consume any shared accumulator from the CalibrationEngine
    # (the engine wires per-expert channels, not whole-module outputs).
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Mandatory — MA-formation detection always runs. There is no flag."""
        return True

    def run(self, ctx: PipelineContext) -> None:
        """Execute the MA-formation detection end-to-end: build calibration
        batches, run the dedicated early-exit pass with decoder-layer +
        MoE-block hooks, detect ``L``.

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
# MA-formation layer detection — module-level private helpers.
# This module is the single source of truth (concern previously known as
# "Phase A" of the pre-refactor Stage 1 monolith; see naming-history note in
# the module docstring).
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

    See the module docstring (Paper / Deviation sections) for the OR rule rationale.
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
