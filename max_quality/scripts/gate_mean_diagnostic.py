"""Gate-mean diagnostic for the Qwen3.5MoE attention output gate.

Purpose
-------
Verifies the empirical justification for the lowered Stage 1 Phase A
``ma_growth_ratio = 3.0`` threshold (see ``max_quality/docs/stage1_review_10.05.2026.txt``
§Q1 / "gating hypothesis").

The Qwen3.5MoE / Qwen3.6 attention block computes::

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2),
        2, dim=-1,
    )
    # ... standard attention ...
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)

The reviewer's hypothesis predicts the residual-stream growth threshold should
be calibrated as::

    threshold_adjusted = threshold_paper * E[sigmoid(gate)]_avg_over_full_attn_layers

For Qwen3.6-35B-A3B the empirical finding ``ma_growth_ratio = 3.0`` (vs. the
paper's 5.0) implies ``E[sigmoid(gate)] ~= 0.6``.  This script computes that
expectation by hooking each ``full_attention`` layer's ``self_attn.q_proj`` and
averaging ``sigmoid(gate)`` over a small calibration sample.

Output
------
A JSON document with::

    {
      "per_layer_E_sigmoid_gate": {<layer_idx>: float, ...},
      "overall_mean": float,
      "predicted_threshold_correction_factor": float,   # == overall_mean
      "predicted_threshold_for_ma_growth_ratio": float, # == 5.0 * overall_mean
      "config": {...}
    }

Invocation
----------
    PYTHONPATH=max_quality/src python max_quality/scripts/gate_mean_diagnostic.py \\
        --num-samples 256 --output /tmp/gate_mean.json

Mirrors the structure of ``phase_a_diagnostic.py``.  Like that script, this
requires GPU + the 35B weights — it's intended for one-shot offline runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_compress.utils.calibration import (
    CalibrationSpec,
    build_calibration_tensor,
    iter_batches,
    spec_from_config,
)
from moe_compress.utils.model_io import iter_decoder_layers

LOG = logging.getLogger("gate_mean_diagnostic")

# Stage 1 Phase A "paper" growth threshold (Qwen3-30B baseline, no output gate).
# We multiply this by the measured E[sigmoid(gate)] to predict the corrected
# threshold for Qwen3.6-35B-A3B.
_MA_GROWTH_RATIO_PAPER = 5.0


def _full_attn_layer_indices(model) -> list[int]:
    """Return decoder layer indices whose `layer_types[i] == "full_attention"`.

    Qwen3.5MoE has a hybrid linear+full-attention layout; only the
    full-attention layers contain the gated `q_proj` we care about.  Linear
    (DeltaNet) layers have a different attention head and no q_proj output
    gate.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("Model has no `config` attribute")
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types:
        raise RuntimeError(
            "Model config has no `layer_types`; cannot identify full-attention "
            "layers. (Required for Qwen3.5MoE hybrid attention layout.)"
        )
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def _capture_gate_means(
    model,
    batches,
    full_attn_indices: list[int],
) -> dict[int, float]:
    """Hook each full-attention layer's `q_proj` and return per-layer
    `E[sigmoid(gate)]` over the calibration set.

    The raw `q_proj(hidden_states)` output is flat with shape
    ``(*input_shape, num_heads * head_dim * 2)``.  The model then ``view``s
    this as ``(*input_shape, num_heads, head_dim * 2)`` and ``chunk``s along
    the last dim — the second half is the gate logits.  We replicate that
    reshape inside the hook (using `head_dim` from the parent attention
    module) so the means are taken over the *correct* gate-half elements,
    not over a per-head-axis split of the flat tensor.
    """
    # idx -> running (sum, count) of sigmoid(gate)
    sums: dict[int, float] = {i: 0.0 for i in full_attn_indices}
    counts: dict[int, int] = {i: 0 for i in full_attn_indices}
    handles: list = []

    decoder_layers = dict(iter_decoder_layers(model))  # idx -> layer module

    def _make_hook(layer_idx: int, head_dim: int):
        # Capture head_dim by closure so the hook can reshape the flat
        # q_proj output to (..., num_heads, head_dim * 2) before splitting.
        def _hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            # output shape: (*input_shape, num_heads * head_dim * 2)
            if output.shape[-1] % (head_dim * 2) != 0:
                LOG.warning(
                    "Layer %d: q_proj out_features=%d not divisible by "
                    "2*head_dim=%d; skipping",
                    layer_idx, output.shape[-1], head_dim * 2,
                )
                return
            input_shape = output.shape[:-1]
            reshaped = output.view(*input_shape, -1, head_dim * 2)
            # second half along last dim is the gate (matches Qwen3_5MoeAttention.forward)
            gate = reshaped[..., head_dim:]
            sg = torch.sigmoid(gate.detach().float())
            sums[layer_idx] += float(sg.sum().item())
            counts[layer_idx] += int(sg.numel())
        return _hook

    for li in full_attn_indices:
        layer = decoder_layers.get(li)
        if layer is None:
            LOG.warning("Layer index %d not found in iter_decoder_layers; skipping", li)
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            LOG.warning("Layer %d has no `self_attn`; skipping", li)
            continue
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is None:
            LOG.warning("Layer %d has no `self_attn.q_proj`; skipping", li)
            continue
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            # fall back to config's head_dim — same source the model uses internally
            text_cfg = getattr(model.config, "text_config", model.config)
            head_dim = getattr(text_cfg, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            LOG.warning(
                "Layer %d: could not determine head_dim (got %r); skipping",
                li, head_dim,
            )
            continue
        handles.append(q_proj.register_forward_hook(_make_hook(li, head_dim)))

    LOG.info("Registered %d hooks on full-attention layers", len(handles))

    try:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(batches):
                batch = batch.to(device)
                model(input_ids=batch)
                if (i + 1) % 32 == 0:
                    LOG.info("gate-mean forward %d/%d", i + 1, len(batches))
    finally:
        for h in handles:
            h.remove()

    means: dict[int, float] = {}
    for li in full_attn_indices:
        n = counts.get(li, 0)
        means[li] = (sums[li] / n) if n > 0 else 0.0
    return means


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B"))
    p.add_argument("--num-samples", type=int,
                   default=int(os.environ.get("NUM_SAMPLES", "256")),
                   help="Calibration sample count. 256 is enough for a stable "
                        "E[sigmoid(gate)]; we don't need a Phase B reservoir here.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--output", type=Path,
                   default=Path(os.environ.get("OUTPUT", "/tmp/gate_mean.json")))
    p.add_argument("--cache-dir", type=Path,
                   default=Path(os.environ.get("CACHE_DIR", "/tmp/gate_mean_cache")))
    p.add_argument("--config", type=Path, default=None,
                   help="Optional pipeline YAML for matching the Stage 1 calibration spec exactly. "
                        "If omitted, falls back to the c4-math-code legacy default.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    LOG.info("=" * 60)
    LOG.info(" Gate-mean diagnostic (one-shot, %d samples)", args.num_samples)
    LOG.info("=" * 60)
    LOG.info("model       = %s", args.model)
    LOG.info("num-samples = %d", args.num_samples)
    LOG.info("seq-len     = %d", args.seq_len)
    LOG.info("dtype       = %s", args.dtype)
    LOG.info("output      = %s", args.output)
    LOG.info("cache-dir   = %s", args.cache_dir)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — refusing to run on CPU (model is 70 GB).")
    LOG.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tokenizer + calibration tensor
    # ------------------------------------------------------------------
    LOG.info("Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.config:
        import yaml  # noqa: PLC0415 — optional dep
        cfg = yaml.safe_load(args.config.read_text())
        cal_cfg = cfg.get("calibration", {})
        spec = spec_from_config(cal_cfg, num_sequences_override=args.num_samples, seed_offset=2)
    else:
        spec = CalibrationSpec(
            source="c4-math-code",
            num_sequences=args.num_samples,
            sequence_length=args.seq_len,
            seed=42,
            domain_mix={"c4": 0.6, "math": 0.2, "code": 0.2},
        )
    LOG.info("Calibration spec: %s", spec)

    calib = build_calibration_tensor(tok, spec, cache_dir=args.cache_dir)
    batches = iter_batches(calib, batch_size=1)
    LOG.info("Built %d calibration batches", len(batches))

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    LOG.info("Loading model in %s …", args.dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    LOG.info("Model loaded in %.1fs", time.time() - t0)

    full_attn_indices = _full_attn_layer_indices(model)
    LOG.info("Full-attention layer indices (%d total): %s",
             len(full_attn_indices), full_attn_indices)

    # ------------------------------------------------------------------
    # 3. Forward pass + gate-mean capture
    # ------------------------------------------------------------------
    LOG.info("Running gate-mean forward pass …")
    t0 = time.time()
    means = _capture_gate_means(model, batches, full_attn_indices)
    LOG.info("Gate-mean capture done in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # 4. Aggregate + write JSON
    # ------------------------------------------------------------------
    overall = sum(means.values()) / max(len(means), 1)
    predicted_threshold = _MA_GROWTH_RATIO_PAPER * overall

    out = {
        "model": args.model,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "per_layer_E_sigmoid_gate": {str(k): v for k, v in sorted(means.items())},
        "overall_mean": overall,
        "predicted_threshold_correction_factor": overall,
        "predicted_threshold_for_ma_growth_ratio": predicted_threshold,
        "ma_growth_ratio_paper": _MA_GROWTH_RATIO_PAPER,
        "full_attn_layer_indices": full_attn_indices,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    LOG.info("Wrote %s (%d bytes)", args.output, args.output.stat().st_size)

    # Pretty-print summary
    print()
    print("=" * 72)
    print("PER-LAYER E[sigmoid(gate)] — Qwen3.5MoE full_attention layers")
    print("=" * 72)
    print(f"{'layer':>6} {'E[sigmoid(gate)]':>20}")
    print("-" * 72)
    for li in sorted(means):
        print(f"{li:>6} {means[li]:>20.4f}")
    print("-" * 72)
    print(f"{'OVERALL':>6} {overall:>20.4f}")
    print()
    print(f"  paper growth threshold (no gate)     : {_MA_GROWTH_RATIO_PAPER:.2f}")
    print(f"  predicted threshold @ this gate-mean : {predicted_threshold:.4f}")
    print(f"    (== {_MA_GROWTH_RATIO_PAPER:.2f} × {overall:.4f})")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
