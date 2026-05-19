"""Precompute Stage-2 teacher MoE-block outputs — the merge-heal target.

Standalone HF Jobs UV script. Loads the original (uncompressed) teacher model
alone — no student in memory, so the full BF16 weights (~70 GB) fit on a single
80 GB+ card — runs the Stage-2 heal calibration set forward once, captures every
MoE block's output hidden-state per layer, and writes a sidecar that Stage 2's
per-layer merge-heal step reads.

Why this exists: the Stage-2 merge-heal fine-tunes each merged layer to
reproduce the teacher's same-layer MoE-block output. Stage 2 itself never loads
a teacher; this precompute supplies the target as a disk artifact (one teacher
forward, reusable across every sweep row).

Calibration: a slice of ``nvidia/Nemotron-Cascade-2-SFT-Data`` drawn with
``seed_offset=2`` (the Stage-2 namespace) — deliberately disjoint from Stage
2.5's KD slice (``seed_offset=5``), so healing and router-KD do not train on the
same tokens.

Output sidecar ``_stage2_teacher_layer_outputs.pt`` (BF16). For a 512k-token
budget × 40 layers × hidden 2048 this is ~84 GB on disk; the script holds the
per-layer buffers in CPU RAM before serializing, so run it on a high-RAM box.

Run it:

    hf jobs uv run hf_jobs/precompute_teacher_layer_outputs.py \\
        --flavor a100-large \\
        --volume hf://buckets/pirola/moe-cache:/mnt/cache \\
        --secrets HF_TOKEN \\
        --timeout 2h
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.5.0,<2.11.0",
#     "transformers>=4.57.0",
#     "accelerate>=1.0.0",
#     "datasets>=3.0.0",
#     "safetensors>=0.4.5",
#     "tokenizers>=0.20.0",
#     "sentencepiece>=0.2.0",
#     "huggingface_hub>=0.26.0",
#     "einops>=0.8.0",
#     "numpy>=1.26.0",
#     "scipy>=1.11.0",
#     "pyyaml>=6.0",
# ]
# ///
#
# NOTE: scipy is required transitively — this script imports
# `moe_compress.stage2_reap_ream` (for `_HEAL_SIDECAR_FORMAT_VERSION`), and that
# module imports `scipy.optimize.linear_sum_assignment`.

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

LOG = logging.getLogger("precompute_teacher_layer_outputs")

# Sidecar schema version: the single source of truth is
# moe_compress.stage2_reap_ream._HEAL_SIDECAR_FORMAT_VERSION (the Stage-2
# consumer validates against it). It is imported in _main() once the package
# is on sys.path so the producer and consumer can never silently drift.

# Calibration seed namespace for Stage 2 — MUST be disjoint from Stage 2.5's
# (seed_offset=5) and MUST match the value Stage 2's merge-heal consumer uses.
STAGE2_SEED_OFFSET = 2


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _parse_args()

    # Bring our package onto sys.path. Mirrors hf_jobs/entrypoint.py logic.
    code_root = Path(args.code_root).expanduser().resolve()
    sys.path.insert(0, str(code_root / "src"))

    from moe_compress.stage2_reap_ream import _HEAL_SIDECAR_FORMAT_VERSION
    from moe_compress.utils.calibration import (
        build_calibration_tensor,
        iter_batches,
        spec_from_config,
    )
    from moe_compress.utils.model_io import iter_moe_layers, load_model

    config_path = Path(args.config).expanduser()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cal = config["calibration"]

    artifacts_dir = Path(args.artifacts_dir).expanduser()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Token budget → (num_sequences, sequence_length). sequence_length always
    # comes from the calibration config — the Stage-2 consumer uses
    # cal["sequence_length"] too, so it must not diverge here. num_sequences is
    # derived with FLOOR division (token_budget // seq_len), exactly matching
    # the Stage-2 consumer (stage2_reap_ream._HealConfig usage); a ceil here
    # would make the two disagree whenever token_budget is not a seq_len
    # multiple, breaking sidecar row-alignment.
    seq_len = int(cal["sequence_length"])
    if seq_len <= 0:
        raise ValueError(f"sequence_length must be > 0, got {seq_len}")
    num_sequences = max(1, int(args.token_budget) // seq_len)
    n_tokens = num_sequences * seq_len
    LOG.info(
        "Heal calibration: token_budget=%d -> %d sequences x %d tokens = %d tokens",
        args.token_budget, num_sequences, seq_len, n_tokens,
    )

    spec = spec_from_config(
        cal,
        num_sequences_override=num_sequences,
        sequence_length_override=seq_len,
        seed_offset=args.seed_offset,
    )

    from transformers import AutoTokenizer
    # The calibration cache identity (and the consumer's validation gate) is
    # keyed on the TOKENIZER, not the teacher-weights repo: a quantized teacher
    # (e.g. -FP8) shares a byte-identical tokenizer with the base model, and the
    # Stage-2 consumer keys its sidecar check on config["model"]["name_or_path"].
    # Use that same id here so --model can point at any teacher checkpoint
    # without spuriously failing the calib_cache_key gate.
    tokenizer_id = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    calib_cache_key = spec.cache_key(tokenizer_id)
    batch_size = int(args.batch_size)
    batches = list(iter_batches(calib, batch_size=batch_size))
    LOG.info("Calibration: %d batches x batch_size %d", len(batches), batch_size)

    # Load teacher alone — full BF16, no offload (no student in memory).
    LOG.info("Loading teacher %s (no student in memory -> full BF16 on cuda)", args.model)
    t_start = time.monotonic()
    teacher, _tok = load_model(
        args.model,
        revision=config["model"]["revision"],
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"],
        load_in_4bit=False,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    teacher.train(False)  # inference mode
    for p in teacher.parameters():
        p.requires_grad_(False)
    LOG.info("Teacher loaded in %.1fs", time.monotonic() - t_start)

    teacher_refs = list(iter_moe_layers(teacher))
    layer_indices = [ref.layer_idx for ref in teacher_refs]
    LOG.info("Teacher: %d MoE layers", len(teacher_refs))

    # Forward hooks on every MoE block. The block returns either the hidden
    # state directly or a tuple whose first element is it (router logits may
    # follow) — mirror _LayerOutputCapture's handling. Each batch contributes a
    # [B*T, hidden] slice; slices are concatenated per layer at the end.
    per_layer_buffers: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}
    handles: list = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            flat = tensor.detach().reshape(-1, tensor.shape[-1])
            per_layer_buffers[layer_idx].append(flat.to(torch.bfloat16).cpu())
        return _hook

    for ref in teacher_refs:
        handles.append(ref.mlp.register_forward_hook(_make_hook(ref.layer_idx)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_batches = len(batches)
    log_every = max(1, n_batches // 50)
    t0 = time.monotonic()
    try:
        with torch.no_grad():
            for i, batch in enumerate(batches):
                teacher(input_ids=batch.to(device))
                if (i + 1) % log_every == 0:
                    elapsed = time.monotonic() - t0
                    eta = elapsed * (n_batches - i - 1) / (i + 1)
                    LOG.info("batch %d/%d | %.1fs elapsed, ~%.0fs ETA",
                             i + 1, n_batches, elapsed, eta)
    finally:
        for h in handles:
            h.remove()

    # Concatenate per-layer buffers; every layer must have exactly n_tokens rows.
    LOG.info("Concatenating per-layer buffers")
    layer_outputs: dict[int, torch.Tensor] = {}
    hidden_size = 0
    total_bytes = 0
    for li in layer_indices:
        parts = per_layer_buffers[li]
        if not parts:
            raise RuntimeError(f"Layer {li}: no MoE-block output captured")
        t = torch.cat(parts, dim=0).contiguous()
        if t.shape[0] != n_tokens:
            raise RuntimeError(
                f"Layer {li}: captured {t.shape[0]} token rows, expected {n_tokens}"
            )
        layer_outputs[li] = t
        hidden_size = t.shape[1]
        total_bytes += t.numel() * t.element_size()
    LOG.info("Total sidecar size: %.2f GB across %d layers", total_bytes / 1e9, len(layer_outputs))

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _HEAL_SIDECAR_FORMAT_VERSION,
        "layer_outputs": layer_outputs,
        "n_tokens": int(n_tokens),
        "hidden_size": int(hidden_size),
        "layer_indices": list(layer_indices),
        "num_sequences": int(spec.num_sequences),
        "sequence_length": int(spec.sequence_length),
        "batch_size": batch_size,
        "model": args.model,
        "calibration_seed_offset": int(args.seed_offset),
        "calib_cache_key": calib_cache_key,
    }
    # Atomic write: torch.save to a .tmp, fsync, then os.replace.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    LOG.info("Saving -> %s", out_path)
    torch.save(payload, tmp_path)
    fd = os.open(str(tmp_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, out_path)
    LOG.info("Saved (%.2f GB on disk)", out_path.stat().st_size / 1e9)

    if args.upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload_repo, repo_type="model", private=True, exist_ok=True)
        LOG.info("Uploading -> https://huggingface.co/%s", args.upload_repo)
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_path.name,
            repo_id=args.upload_repo,
            repo_type="model",
        )
        LOG.info("Upload complete")

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Stage-2 teacher MoE-block outputs")
    p.add_argument("--config", default="/mnt/cache/code/configs/qwen36_35b_a3b_30pct.yaml")
    p.add_argument("--code-root", default="/mnt/cache/code")
    p.add_argument("--model", default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B"))
    p.add_argument("--artifacts-dir", default="/mnt/cache/artifacts")
    p.add_argument("--output", default="/mnt/cache/artifacts/_stage2_teacher_layer_outputs.pt")
    p.add_argument(
        "--token-budget", type=int, default=524288,
        help="Heal calibration token budget (num_sequences derived from it via "
             "floor division by the calibration config's sequence_length).",
    )
    p.add_argument(
        "--seed-offset", type=int, default=STAGE2_SEED_OFFSET,
        help="Calibration seed namespace; must be disjoint from Stage 2.5's (5).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--upload-repo",
        default=os.environ.get("TEACHER_LAYER_OUTPUTS_REPO", ""),
        help="Hub model repo to upload the ~84 GB sidecar to. OPT-IN: defaults "
        "to empty (skip upload) so a bare run never pushes a huge artifact "
        "unexpectedly; pass a repo (or set TEACHER_LAYER_OUTPUTS_REPO) to enable.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(_main())
