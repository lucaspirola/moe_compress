"""L31 reset-hypothesis verification on Qwen3.6-35B-A3B.

Purpose
-------
Verifies the reviewer's hypothesis (see ``max_quality/docs/stage1_review_10.05.2026.txt``
§Q3) that the residual-stream collapse observed at decoder layer 31 (the 8th
full-attention layer in the [linear, linear, linear, full]-cycle) is gate-mediated
denoising — i.e. ``sigmoid(gate)`` is learned to be near-zero in the feature
dimensions that carry massive activations (MAs), selectively suppressing the
attention branch contribution in those channels while the MoE branch operates
normally.

What it does
------------
For each calibration batch:

1. Hook L30's decoder-layer output (= the residual stream entering L31) and
   identify the top-K feature-dimension indices by mean absolute magnitude over
   batch+token axes.  These are the *MA-carrying dimensions* for the batch.

2. Hook L31's ``self_attn.q_proj`` output and extract ``sigmoid(gate)`` using
   the SAME ``view(*, num_heads, head_dim*2)`` + chunk-along-last-dim layout
   used in ``gate_mean_diagnostic.py`` (verified against the
   ``Qwen3_5MoeAttention.forward`` source — flat slicing is wrong because the
   per-head dim is interleaved).

3. Hook L31's ``self_attn.o_proj`` output — the gated attention contribution
   that gets added to the residual.

4. Hook L31's ``mlp`` output — the MoE contribution that gets added to the
   residual.

After each forward, reduce each of (2)/(3)/(4) over batch+token to a per-feature
mean magnitude and partition into MA-dim and non-MA-dim groups.  The script
reports the mean of each partition averaged across batches.

Expected outcome (if hypothesis holds)
--------------------------------------
* ``L31_sigmoid_gate_in_ma_dims / L31_sigmoid_gate_in_non_ma_dims`` ≪ 1.0
* ``L31_attn_output_in_ma_dims  / L31_attn_output_in_non_ma_dims``  ≪ 1.0
* ``L31_moe_output_in_ma_dims   / L31_moe_output_in_non_ma_dims``   ~ 1.0
  (MoE branch is not selectively suppressed in MA dims)

Output
------
A JSON document — see ``--output``.

Invocation
----------
    PYTHONPATH=max_quality/src python max_quality/scripts/l31_verification.py \\
        --num-samples 256 --output /tmp/l31_verification.json

Requires GPU + the 35B weights — intended for one-shot offline runs.  Mirrors
``gate_mean_diagnostic.py``.
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

LOG = logging.getLogger("l31_verification")


def _layer_types(model) -> list[str]:
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
    return list(layer_types)


def _resolve_head_dim(target_layer_module, model) -> int | None:
    """Same fallback path as gate_mean_diagnostic.py."""
    attn = getattr(target_layer_module, "self_attn", None)
    head_dim = getattr(attn, "head_dim", None) if attn is not None else None
    if not isinstance(head_dim, int) or head_dim <= 0:
        text_cfg = getattr(model.config, "text_config", model.config)
        head_dim = getattr(text_cfg, "head_dim", None)
    if not isinstance(head_dim, int) or head_dim <= 0:
        return None
    return head_dim


def _run_verification(
    model,
    batches,
    target_layer: int,
    ma_source_layer: int,
    ma_dim_count_K: int,
) -> dict:
    """Run forward passes and return aggregated diagnostics."""
    decoder_layers = dict(iter_decoder_layers(model))
    if ma_source_layer not in decoder_layers:
        raise RuntimeError(f"MA-source layer {ma_source_layer} not found")
    if target_layer not in decoder_layers:
        raise RuntimeError(f"Target layer {target_layer} not found")

    l_source = decoder_layers[ma_source_layer]
    l_target = decoder_layers[target_layer]

    attn = getattr(l_target, "self_attn", None)
    if attn is None:
        raise RuntimeError(f"Target layer {target_layer} has no `self_attn`")
    q_proj = getattr(attn, "q_proj", None)
    o_proj = getattr(attn, "o_proj", None)
    mlp = getattr(l_target, "mlp", None)
    if q_proj is None or o_proj is None or mlp is None:
        raise RuntimeError(
            f"Target layer {target_layer} missing one of "
            f"self_attn.q_proj/self_attn.o_proj/mlp"
        )

    head_dim = _resolve_head_dim(l_target, model)
    if head_dim is None:
        raise RuntimeError(
            f"Could not determine head_dim for target layer {target_layer}"
        )
    LOG.info("Resolved head_dim=%d for L%d", head_dim, target_layer)

    # Per-batch state — reset on each forward.
    batch_state: dict = {
        "ma_dims": None,                # set[int]
        "L30_per_feat": None,           # tensor (hidden,)
        "L31_sigmoid_gate": None,       # tensor (B, T, hidden)
        "L31_attn_output": None,        # tensor (B, T, hidden)
        "L31_moe_output": None,         # tensor (B, T, hidden)
    }

    # ------------------------------------------------------------------
    # Hook definitions
    # ------------------------------------------------------------------
    def _l30_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor):
            return
        # h shape: (B, T, hidden). Mean abs over (B, T) -> (hidden,)
        per_feat = h.detach().abs().float().mean(dim=tuple(range(h.dim() - 1)))
        K = min(ma_dim_count_K, per_feat.numel())
        topk_idx = torch.topk(per_feat, K).indices.cpu().tolist()
        batch_state["ma_dims"] = set(int(i) for i in topk_idx)
        batch_state["L30_per_feat"] = per_feat.detach().cpu()

    def _l31_qproj_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        # Same view-and-chunk pattern as gate_mean_diagnostic.py.
        if output.shape[-1] % (head_dim * 2) != 0:
            LOG.warning(
                "L%d q_proj out_features=%d not divisible by 2*head_dim=%d; "
                "skipping gate capture",
                target_layer, output.shape[-1], head_dim * 2,
            )
            return
        input_shape = output.shape[:-1]
        viewed = output.view(*input_shape, -1, head_dim * 2)
        gate = viewed[..., head_dim:]                     # (..., num_heads, head_dim)
        sg = torch.sigmoid(gate.detach().float())
        # Flatten the (num_heads, head_dim) tail back to feature space.
        sg_flat = sg.reshape(*input_shape, -1)            # (B, T, num_heads*head_dim)
        batch_state["L31_sigmoid_gate"] = sg_flat.cpu()

    def _l31_oproj_hook(_module, _inputs, output):
        # o_proj output is the post-gated attention contribution that is added
        # to the residual stream — shape (B, T, hidden).
        if not isinstance(output, torch.Tensor):
            return
        batch_state["L31_attn_output"] = output.detach().abs().float().cpu()

    def _l31_mlp_hook(_module, _inputs, output):
        # Qwen3.5MoE's MoeSparseMlp returns the post-routing-weighted-sum
        # tensor (sometimes wrapped in a tuple).
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor):
            return
        batch_state["L31_moe_output"] = h.detach().abs().float().cpu()

    # ------------------------------------------------------------------
    # Aggregation across batches
    # ------------------------------------------------------------------
    agg_sums: dict[str, float] = {
        "sigmoid_gate_ma": 0.0, "sigmoid_gate_non_ma": 0.0,
        "attn_output_ma": 0.0, "attn_output_non_ma": 0.0,
        "moe_output_ma": 0.0, "moe_output_non_ma": 0.0,
    }
    agg_counts: dict[str, int] = {k: 0 for k in agg_sums}
    ma_dim_history: list[set[int]] = []
    l30_per_feat_accum: torch.Tensor | None = None
    l30_per_feat_count: int = 0

    def _accumulate_after_batch():
        sg = batch_state["L31_sigmoid_gate"]
        ao = batch_state["L31_attn_output"]
        mo = batch_state["L31_moe_output"]
        ma_dims = batch_state["ma_dims"]
        if sg is None or ao is None or mo is None or ma_dims is None:
            LOG.warning("Batch missing one of (sg, ao, mo, ma_dims); skipping accumulate")
            return
        # Determine hidden size from o_proj output (this IS hidden-dim space).
        hidden = ao.shape[-1]
        # If sigmoid(gate) doesn't share hidden width (e.g. GQA with num_kv_heads
        # collapsed inside attention so q_proj is num_heads*head_dim), we partition
        # it on its own width.  For the partition we still use the same MA-dim set
        # selected from the residual stream's hidden space — the q_proj output
        # width must match for the comparison to be meaningful.  Warn if not.
        ma_idx_sorted = sorted(ma_dims)
        ma_idx_t = torch.tensor(ma_idx_sorted, dtype=torch.long)

        for name, tensor in (
            ("sigmoid_gate", sg),
            ("attn_output", ao),
            ("moe_output", mo),
        ):
            if tensor.shape[-1] != hidden:
                LOG.warning(
                    "Batch: %s feature dim %d != residual hidden %d; "
                    "partitioning over %s's own feature axis instead",
                    name, tensor.shape[-1], hidden, name,
                )
            feat = tensor.shape[-1]
            local_ma = ma_idx_t[ma_idx_t < feat]
            non_ma_idx = torch.tensor(
                [i for i in range(feat) if i not in ma_dims],
                dtype=torch.long,
            )
            per_feat = tensor.mean(dim=tuple(range(tensor.dim() - 1)))  # (feat,)
            if len(local_ma) > 0:
                ma_mean = float(per_feat[local_ma].mean().item())
                agg_sums[f"{name}_ma"] += ma_mean
                agg_counts[f"{name}_ma"] += 1
            if len(non_ma_idx) > 0:
                non_mean = float(per_feat[non_ma_idx].mean().item())
                agg_sums[f"{name}_non_ma"] += non_mean
                agg_counts[f"{name}_non_ma"] += 1

    # ------------------------------------------------------------------
    # Register hooks + run
    # ------------------------------------------------------------------
    handles = [
        l_source.register_forward_hook(_l30_hook),
        q_proj.register_forward_hook(_l31_qproj_hook),
        o_proj.register_forward_hook(_l31_oproj_hook),
        mlp.register_forward_hook(_l31_mlp_hook),
    ]
    LOG.info("Registered %d hooks (L%d residual + L%d q/o/mlp)",
             len(handles), ma_source_layer, target_layer)

    try:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(batches):
                # Reset per-batch state
                for key in ("ma_dims", "L30_per_feat", "L31_sigmoid_gate",
                            "L31_attn_output", "L31_moe_output"):
                    batch_state[key] = None
                batch = batch.to(device)
                model(input_ids=batch)
                ma_dim_history.append(
                    set(batch_state["ma_dims"]) if batch_state["ma_dims"] else set()
                )
                if batch_state["L30_per_feat"] is not None:
                    pf = batch_state["L30_per_feat"]
                    if l30_per_feat_accum is None:
                        l30_per_feat_accum = pf.clone()
                    else:
                        l30_per_feat_accum = l30_per_feat_accum + pf
                    l30_per_feat_count += 1
                _accumulate_after_batch()
                if (i + 1) % 32 == 0:
                    LOG.info("L31-verify forward %d/%d", i + 1, len(batches))
    finally:
        for h in handles:
            h.remove()

    # ------------------------------------------------------------------
    # Build aggregated result
    # ------------------------------------------------------------------
    def _avg(name: str) -> float:
        n = agg_counts[name]
        return (agg_sums[name] / n) if n > 0 else 0.0

    diagnostics = {
        "L31_sigmoid_gate_in_ma_dims": _avg("sigmoid_gate_ma"),
        "L31_sigmoid_gate_in_non_ma_dims": _avg("sigmoid_gate_non_ma"),
        "L31_attn_output_in_ma_dims": _avg("attn_output_ma"),
        "L31_attn_output_in_non_ma_dims": _avg("attn_output_non_ma"),
        "L31_moe_output_in_ma_dims": _avg("moe_output_ma"),
        "L31_moe_output_in_non_ma_dims": _avg("moe_output_non_ma"),
    }

    ma_dim_union: set[int] = set().union(*ma_dim_history) if ma_dim_history else set()
    # Per-MA-dim mean abs L30 residual magnitude (averaged across batches).
    l30_max_per_dim: dict[str, float] = {}
    if l30_per_feat_accum is not None and l30_per_feat_count > 0:
        mean_per_feat = l30_per_feat_accum / l30_per_feat_count
        for d in sorted(ma_dim_union):
            if d < mean_per_feat.numel():
                l30_max_per_dim[str(d)] = float(mean_per_feat[d].item())

    eps = 1e-12
    interpretation = {
        "gate_suppression_ratio_ma_over_non_ma": (
            diagnostics["L31_sigmoid_gate_in_ma_dims"]
            / max(diagnostics["L31_sigmoid_gate_in_non_ma_dims"], eps)
        ),
        "attn_suppression_ratio_ma_over_non_ma": (
            diagnostics["L31_attn_output_in_ma_dims"]
            / max(diagnostics["L31_attn_output_in_non_ma_dims"], eps)
        ),
        "moe_ratio_ma_over_non_ma": (
            diagnostics["L31_moe_output_in_ma_dims"]
            / max(diagnostics["L31_moe_output_in_non_ma_dims"], eps)
        ),
        "expected": (
            "If hypothesis holds: gate_suppression_ratio << 1.0, "
            "attn_suppression_ratio << 1.0, moe_ratio ~= 1.0"
        ),
    }

    return {
        "diagnostics": diagnostics,
        "interpretation": interpretation,
        "ma_carrying_dimensions_union": sorted(ma_dim_union),
        "ma_dim_union_size": len(ma_dim_union),
        "ma_dim_history_per_batch": [sorted(s) for s in ma_dim_history],
        "L30_residual_mean_abs_per_dim": l30_max_per_dim,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model",
                   default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B"))
    p.add_argument("--num-samples", type=int,
                   default=int(os.environ.get("NUM_SAMPLES", "256")),
                   help="Calibration sample count.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--output", type=Path,
                   default=Path(os.environ.get("OUTPUT", "/tmp/l31_verification.json")))
    p.add_argument("--cache-dir", type=Path,
                   default=Path(os.environ.get("CACHE_DIR", "/tmp/l31_verification_cache")))
    p.add_argument("--config", type=Path, default=None,
                   help="Optional pipeline YAML to match Stage 1 calibration spec exactly. "
                        "If omitted, falls back to the c4-math-code legacy default.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--target-layer", type=int, default=31,
                   help="Decoder layer to verify the gate-mediated reset on (default: 31).")
    p.add_argument("--ma-source-layer", type=int, default=30,
                   help="Decoder layer whose residual output is used to identify "
                        "MA-carrying feature dimensions (default: 30).")
    p.add_argument("--ma-dim-count-K", type=int, default=16,
                   help="K for top-K MA-dimension selection at the source layer (default: 16).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    LOG.info("=" * 60)
    LOG.info(" L31 reset-hypothesis verification (one-shot, %d samples)",
             args.num_samples)
    LOG.info("=" * 60)
    LOG.info("model            = %s", args.model)
    LOG.info("num-samples      = %d", args.num_samples)
    LOG.info("seq-len          = %d", args.seq_len)
    LOG.info("dtype            = %s", args.dtype)
    LOG.info("target-layer     = %d", args.target_layer)
    LOG.info("ma-source-layer  = %d", args.ma_source_layer)
    LOG.info("ma-dim-count-K   = %d", args.ma_dim_count_K)
    LOG.info("output           = %s", args.output)
    LOG.info("cache-dir        = %s", args.cache_dir)

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
        spec = spec_from_config(cal_cfg, num_sequences_override=args.num_samples, seed_offset=3)
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

    # ------------------------------------------------------------------
    # 3. Validate that target_layer is a full_attention layer
    # ------------------------------------------------------------------
    layer_types = _layer_types(model)
    if args.target_layer >= len(layer_types):
        LOG.error("target-layer %d is out of range (model has %d layers); aborting.",
                  args.target_layer, len(layer_types))
        return 2
    if layer_types[args.target_layer] != "full_attention":
        LOG.error(
            "target-layer %d has layer_types[%d]=%r — not a full_attention layer. "
            "The L31 reset hypothesis is specific to gated full-attention layers; "
            "aborting.",
            args.target_layer, args.target_layer, layer_types[args.target_layer],
        )
        return 2
    if args.ma_source_layer < 0 or args.ma_source_layer >= len(layer_types):
        LOG.error("ma-source-layer %d is out of range (model has %d layers); aborting.",
                  args.ma_source_layer, len(layer_types))
        return 2

    LOG.info("Validated: layer_types[%d]=%r (full_attention).",
             args.target_layer, layer_types[args.target_layer])
    LOG.info("MA-source layer_types[%d]=%r",
             args.ma_source_layer, layer_types[args.ma_source_layer])

    # ------------------------------------------------------------------
    # 4. Run
    # ------------------------------------------------------------------
    LOG.info("Running L31-verification forward pass …")
    t0 = time.time()
    result_core = _run_verification(
        model,
        batches,
        target_layer=args.target_layer,
        ma_source_layer=args.ma_source_layer,
        ma_dim_count_K=args.ma_dim_count_K,
    )
    LOG.info("Verification done in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # 5. Aggregate + write JSON
    # ------------------------------------------------------------------
    out = {
        "config": {
            "model": args.model,
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "target_layer": args.target_layer,
            "ma_source_layer": args.ma_source_layer,
            "ma_dim_count_K": args.ma_dim_count_K,
            "target_layer_type": layer_types[args.target_layer],
            "ma_source_layer_type": layer_types[args.ma_source_layer],
        },
        **result_core,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    LOG.info("Wrote %s (%d bytes)", args.output, args.output.stat().st_size)

    # Pretty-print summary
    diag = result_core["diagnostics"]
    interp = result_core["interpretation"]
    print()
    print("=" * 72)
    print(" L31 reset-hypothesis — diagnostics")
    print("=" * 72)
    print(f"  target layer   : L{args.target_layer} ({layer_types[args.target_layer]})")
    print(f"  MA source      : L{args.ma_source_layer} (top-{args.ma_dim_count_K} dims by mean|·|)")
    print(f"  MA-dim union   : {len(result_core['ma_carrying_dimensions_union'])} dims "
          f"across {args.num_samples} batches")
    print("-" * 72)
    print(f"  sigmoid(gate)  in MA dims : {diag['L31_sigmoid_gate_in_ma_dims']:.6f}")
    print(f"  sigmoid(gate)  non-MA     : {diag['L31_sigmoid_gate_in_non_ma_dims']:.6f}")
    print(f"  attn_output    in MA dims : {diag['L31_attn_output_in_ma_dims']:.6f}")
    print(f"  attn_output    non-MA     : {diag['L31_attn_output_in_non_ma_dims']:.6f}")
    print(f"  moe_output     in MA dims : {diag['L31_moe_output_in_ma_dims']:.6f}")
    print(f"  moe_output     non-MA     : {diag['L31_moe_output_in_non_ma_dims']:.6f}")
    print("-" * 72)
    print(f"  gate suppression ratio    : {interp['gate_suppression_ratio_ma_over_non_ma']:.4f}  (expect <<1)")
    print(f"  attn suppression ratio    : {interp['attn_suppression_ratio_ma_over_non_ma']:.4f}  (expect <<1)")
    print(f"  moe   ratio               : {interp['moe_ratio_ma_over_non_ma']:.4f}  (expect ~1)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
