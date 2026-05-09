"""Standalone Phase A MA-formation detector diagnostic.

Re-runs ONLY Phase A (the MA-formation layer detection) on a small
calibration sample, and dumps `layer_max[i]` for **every decoder layer**
(MoE and non-MoE) so we can:

  1. Verify whether the production Stage 1's L = {10} result is real.
  2. Identify any MA-formation events the production thresholds missed.
  3. Recommend new values for `ma_growth_ratio` / `ma_ratio`.

This was written because the Stage 1 run at H200 job 69feb306317220dbbd1a7044
produced suspicious results: L = {10} with no apparent magnitude formation
event in the per-expert data, while L39E149 fired at magnitude 302 (207×
the in-L a_max). The hypothesis is that Qwen3.6-35B-A3B's gated attention
(`attn_output_gate=True`) + hybrid linear/full attention layout dampens the
residual-stream signal that Phase A measures, miscalibrating the
`ma_growth_ratio = 5.0` threshold (which was tuned on Qwen3-30B without
output gating).

Mirrors `_detect_ma_layers` in `stage1_grape.py` exactly, but exposes the
`layer_max` dict (currently discarded) and tries multiple thresholds offline
in a single forward pass.

Invocation:
    PYTHONPATH=max_quality/src python max_quality/scripts/phase_a_diagnostic.py \
        --num-samples 256 --output /tmp/phase_a_diag.json

For HF Jobs, see hf_jobs/entrypoint_phase_a_diagnostic.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_compress.utils.calibration import (
    CalibrationSpec,
    build_calibration_tensor,
    iter_batches,
    spec_from_config,
)
from moe_compress.utils.model_io import iter_decoder_layers, iter_moe_layers

LOG = logging.getLogger("phase_a_diagnostic")

# Production thresholds (mirror stage1_grape.py:45-46) — for cross-reference.
_MA_RATIO = 100.0
_MA_GROWTH_RATIO = 5.0


def _detect_with_layer_max_capture(
    model,
    batches,
    moe_layer_indices: list[int],
):
    """Mirror of stage1_grape._detect_ma_layers but RETURN layer_max + first-layer Q99.

    Hooks every decoder layer (MoE and non-MoE) and captures the cumulative
    cross-batch max of |H_l| per layer. Also buffers the first-MoE-layer's
    flattened magnitudes for an exact Q99 (matching the production code path).
    """
    decoder_layers = list(iter_decoder_layers(model))
    if not decoder_layers:
        raise RuntimeError("No decoder layers found via iter_decoder_layers()")
    decoder_layer_modules = {layer: idx for idx, layer in decoder_layers}
    sorted_decoder_indices = sorted(decoder_layer_modules.values())

    moe_layer_set = set(moe_layer_indices)
    first_moe_layer_idx = sorted(moe_layer_indices)[0] if moe_layer_indices else None

    layer_max: dict[int, float] = {idx: 0.0 for idx in sorted_decoder_indices}
    first_layer_q99_buffer: list[np.ndarray] = []
    handles = []

    def _make_hook(layer_idx: int):
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

    for module, layer_idx in decoder_layer_modules.items():
        handles.append(module.register_forward_hook(_make_hook(layer_idx)))

    try:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(batches):
                batch = batch.to(device)
                model(input_ids=batch)
                if (i + 1) % 32 == 0:
                    LOG.info("phase_a forward %d/%d", i + 1, len(batches))
    finally:
        for h in handles:
            h.remove()

    first_layer_q99 = (
        float(np.percentile(np.concatenate(first_layer_q99_buffer), 99.0))
        if first_layer_q99_buffer else 0.0
    )

    return {
        "layer_max": layer_max,
        "sorted_decoder_indices": sorted_decoder_indices,
        "first_moe_layer_idx": first_moe_layer_idx,
        "moe_layer_set": sorted(moe_layer_set),
        "first_layer_q99": first_layer_q99,
    }


def _evaluate_thresholds(layer_max, sorted_decoder_indices, first_moe_layer_idx,
                        moe_layer_set, first_layer_q99,
                        ma_ratio_choices: list[float],
                        ma_growth_ratio_choices: list[float]):
    """Apply the Phase A criteria at each (ma_ratio, ma_growth_ratio) pair and
    return the resulting candidate L for each. Lets us see how sensitive the
    L set is to threshold choice without re-running the forward pass.
    """
    decoder_index_pos = {idx: pos for pos, idx in enumerate(sorted_decoder_indices)}
    out = {}
    for ma_ratio in ma_ratio_choices:
        for ma_growth_ratio in ma_growth_ratio_choices:
            L = []
            for layer_idx in sorted(moe_layer_set):
                if layer_idx == first_moe_layer_idx:
                    if first_layer_q99 > 0 and layer_max[layer_idx] > ma_ratio * first_layer_q99:
                        L.append(layer_idx)
                else:
                    pos = decoder_index_pos[layer_idx]
                    if pos == 0:
                        continue
                    prev_idx = sorted_decoder_indices[pos - 1]
                    prev_max = layer_max[prev_idx]
                    if prev_max > 0 and layer_max[layer_idx] / prev_max > ma_growth_ratio:
                        L.append(layer_idx)
            out[f"ma_ratio={ma_ratio},growth={ma_growth_ratio}"] = L
    return out


def _print_summary_table(result):
    """Pretty-print the per-decoder-layer table."""
    layer_max = result["layer_max"]
    sorted_idx = result["sorted_decoder_indices"]
    moe_set = set(result["moe_layer_set"])
    first_moe = result["first_moe_layer_idx"]
    first_q99 = result["first_layer_q99"]

    print()
    print("=" * 88)
    print("PER-DECODER-LAYER ACTIVATION PROFILE (Phase A signal — residual stream max)")
    print("=" * 88)
    print(f"{'L':>3} {'is_moe':>7} {'max|H_l|':>12} {'growth':>10} {'flag@5x':>9} {'flag@3x':>9} {'flag@2.5x':>11}")
    print("-" * 88)
    prev_max = None
    for li in sorted_idx:
        is_moe = li in moe_set
        max_v = layer_max[li]
        growth = max_v / prev_max if prev_max and prev_max > 0 else None
        gs = f"{growth:>10.2f}" if growth is not None else f"{'—':>10}"
        f5 = "✓" if (growth is not None and growth > 5.0) else ""
        f3 = "✓" if (growth is not None and growth > 3.0) else ""
        f25 = "✓" if (growth is not None and growth > 2.5) else ""
        if li == first_moe and first_q99 > 0:
            ratio = max_v / first_q99
            f5 = f"abs={ratio:.1f}x"
            f3 = ""
            f25 = ""
        print(f"{li:>3} {('moe' if is_moe else 'attn'):>7} {max_v:>12.4g} {gs} {f5:>9} {f3:>9} {f25:>11}")
        prev_max = max_v
    print()
    print(f"first_moe_layer_idx = {first_moe}, first_layer_q99 = {first_q99:.4g}")
    print(f"  (prod thresholds: _MA_RATIO={_MA_RATIO}, _MA_GROWTH_RATIO={_MA_GROWTH_RATIO})")
    print()
    print("=" * 88)
    print("CANDIDATE L SETS AT VARIOUS THRESHOLDS")
    print("=" * 88)
    for label, L in result["candidate_L_sets"].items():
        print(f"  {label:<40}  L = {L}  (size {len(L)})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B"))
    p.add_argument("--num-samples", type=int,
                   default=int(os.environ.get("NUM_SAMPLES", "256")),
                   help="Calibration sample count. 256 is enough for layer_max; "
                        "production used 4000 because Phase B's reservoir benefits.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--output", type=Path,
                   default=Path(os.environ.get("OUTPUT", "/tmp/phase_a_diagnostic.json")))
    p.add_argument("--cache-dir", type=Path,
                   default=Path(os.environ.get("CACHE_DIR", "/tmp/phase_a_cache")))
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
    LOG.info(" Phase A diagnostic (one-shot, ~%d samples, no Phase B/C/D/E)", args.num_samples)
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
        spec = spec_from_config(cal_cfg, num_sequences_override=args.num_samples, seed_offset=1)
    else:
        # Default for the diagnostic: a small c4-math-code spec matching Stage 1's defaults.
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

    moe_layers = list(iter_moe_layers(model))
    moe_layer_indices = [ref.layer_idx for ref in moe_layers]
    LOG.info("Discovered %d MoE layers: %s …", len(moe_layer_indices), moe_layer_indices[:5])

    # ------------------------------------------------------------------
    # 3. Phase A forward + layer_max capture
    # ------------------------------------------------------------------
    LOG.info("Running Phase A forward pass …")
    t0 = time.time()
    raw = _detect_with_layer_max_capture(model, batches, moe_layer_indices)
    LOG.info("Phase A done in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # 4. Evaluate at multiple thresholds
    # ------------------------------------------------------------------
    candidates = _evaluate_thresholds(
        raw["layer_max"],
        raw["sorted_decoder_indices"],
        raw["first_moe_layer_idx"],
        set(raw["moe_layer_set"]),
        raw["first_layer_q99"],
        ma_ratio_choices=[100.0, 50.0, 20.0],
        ma_growth_ratio_choices=[5.0, 4.0, 3.0, 2.5, 2.0],
    )

    # Keep int keys in `result` for the print path; convert to str only when
    # JSON-serializing. Earlier version JSON-stringified up front and the
    # print helper KeyError'd on int lookups against str-keyed dict.
    result = {
        "model": args.model,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "layer_max": raw["layer_max"],
        "sorted_decoder_indices": raw["sorted_decoder_indices"],
        "first_moe_layer_idx": raw["first_moe_layer_idx"],
        "moe_layer_set": raw["moe_layer_set"],
        "first_layer_q99": raw["first_layer_q99"],
        "candidate_L_sets": candidates,
        "production_thresholds": {"ma_ratio": _MA_RATIO, "ma_growth_ratio": _MA_GROWTH_RATIO},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {**result, "layer_max": {str(k): v for k, v in result["layer_max"].items()}}
    args.output.write_text(json.dumps(serializable, indent=2))
    LOG.info("Wrote %s (%d bytes)", args.output, args.output.stat().st_size)

    _print_summary_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
