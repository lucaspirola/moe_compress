"""Precompute Stage 5 teacher router logits — Path B for KD.

Standalone HF Jobs UV script. Loads the original (uncompressed) model
alone (no student → fits cleanly on a single A100 in BF16), runs the
Stage 5 calibration set forward, captures pre-softmax router logits per
layer, and writes a sidecar that Stage 5 can read instead of running the
teacher live.

Why this exists: Stage 5 normally holds teacher (~70 GB) + student
(~50 GB) = ~120 GB on cuda → triggers CPU offload (5–10× slowdown).
Path A (4-bit teacher, ~17 GB) avoids offload but at small KD-signal
precision cost. Path B (this script) gives bit-exact teacher logits
at the cost of a single ~45-min precompute run.

Output sidecar: ``_stage5_teacher_logits.pt`` (BF16). Saved in the
mounted bucket's ``artifacts/`` dir by default; pass ``--output`` to
override. Optionally uploads to a Hub repo via ``--upload-repo`` so a
Stage-5-only HF Job (which doesn't share the bucket from precompute)
can pull it.

Run it:

    hf jobs uv run hf_jobs/precompute_teacher_logits.py \
        --flavor a100-large \
        --volume hf://buckets/pirola/moe-cache:/mnt/cache \
        --secrets HF_TOKEN \
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
#     "pyyaml>=6.0",
# ]
# ///

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

LOG = logging.getLogger("precompute_teacher_logits")


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _parse_args()

    # Bring our package onto sys.path. Mirrors hf_jobs/entrypoint.py logic.
    code_root = Path(args.code_root).expanduser().resolve()
    sys.path.insert(0, str(code_root / "src"))

    from moe_compress.utils.activation_hooks import capture_router_outputs, run_calibration
    from moe_compress.utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
    from moe_compress.utils.model_io import iter_moe_layers, load_model

    # Load config (the same one Stage 5 uses).
    config_path = Path(args.config).expanduser()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    s5 = config["stage5_router_kd"]
    cal = config["calibration"]

    # Build calibration tensor identically to how Stage 5 will, so the
    # logits cache is keyed token-by-token to the same iter_batches.
    artifacts_dir = Path(args.artifacts_dir).expanduser()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    spec = spec_from_config(
        cal,
        num_sequences_override=s5["max_calibration_samples"],
        sequence_length_override=s5["max_sequence_length"],
        seed_offset=5,                            # MUST match stage5_router_kd.run
    )
    LOG.info("Building calibration tensor (%d samples × %d tokens)",
             spec.num_sequences, spec.sequence_length)
    # We don't have the model yet → use AutoTokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batch_size = int(s5["batch_size"])
    batches = list(iter_batches(calib, batch_size=batch_size))
    LOG.info("Calibration: %d batches × batch_size %d", len(batches), batch_size)

    # Load teacher alone — full BF16, no offload (no student in memory).
    LOG.info("Loading teacher %s (no student in memory → full BF16 on cuda)", args.model)
    t_start = time.monotonic()
    teacher, _tok = load_model(
        args.model,
        revision=config["model"]["revision"],
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"],
        load_in_4bit=False,                       # Path B = full precision
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    LOG.info("Teacher loaded in %.1fs", time.monotonic() - t_start)

    teacher_refs = list(iter_moe_layers(teacher))
    LOG.info("Teacher: %d MoE layers", len(teacher_refs))

    # Per-layer accumulator. Each batch contributes a [B*T, num_experts]
    # tensor; we concatenate across batches into one [N_total_tokens,
    # num_experts] tensor per layer in BF16 on CPU.
    per_layer_buffers: dict[int, list[torch.Tensor]] = {ref.layer_idx: [] for ref in teacher_refs}

    # ``device_map="auto"`` already placed teacher tensors. Calling
    # ``model.to(cuda)`` afterwards is redundant and can raise on
    # accelerate-managed (cpu/disk-spilled) models. The 35 B BF16 weights
    # (~70 GB) fit on a single 80 GB A100 alone with no_grad activations
    # at batch=4 (~5–8 GB). If that's tight, run this script with
    # ``--flavor a100x4`` (more cuda) or set ``load_in_4bit=True`` here
    # — but doing so defeats Path B's purpose (we want exact logits).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_batches = len(batches)
    log_every = max(1, n_batches // 50)
    t0 = time.monotonic()
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch_dev = batch.to(device)
            with capture_router_outputs(teacher_refs) as t_out:
                teacher(input_ids=batch_dev)
            for li, logits_list in t_out.items():
                if not logits_list:
                    continue
                # Each forward fires the hook once → list has 1 entry of
                # shape [B*T, num_experts]. Move to CPU bf16 for storage.
                per_layer_buffers[li].append(logits_list[-1].to(torch.bfloat16).cpu())
            if (i + 1) % log_every == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed * (n_batches - i - 1) / (i + 1)
                LOG.info("batch %d/%d | %.1fs elapsed, ~%.0fs ETA",
                         i + 1, n_batches, elapsed, eta)

    # Concatenate per-layer buffers and serialize.
    LOG.info("Concatenating per-layer buffers")
    per_layer: dict[int, torch.Tensor] = {}
    total_bytes = 0
    for li, parts in per_layer_buffers.items():
        if not parts:
            continue
        t = torch.cat(parts, dim=0).contiguous()
        per_layer[li] = t
        total_bytes += t.numel() * t.element_size()
    LOG.info("Total cache size: %.2f GB across %d layers", total_bytes / 1e9, len(per_layer))

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": per_layer,
        "num_samples": int(spec.num_sequences),
        "sequence_length": int(spec.sequence_length),
        "batch_size": batch_size,
        "model": args.model,
        "calibration_seed_offset": 5,
        "format_version": 1,
    }
    LOG.info("Saving → %s", out_path)
    # F-RK-1 fix: previously bare torch.save(payload, out_path) — no
    # tmp+rename, no fsync, no manifest. On HF Jobs pod eviction mid-write
    # (~30 GB), the .pt was truncated at the final path. Stage 5's
    # mmap=True read silently returned zeros/garbage past EOF → degenerate
    # KD signal → hours of Stage 5 training produced a silently-worse
    # student. Now: atomic_torch_save + manifest-last so the Stage 5
    # reader keys on the manifest's existence + size match.
    from moe_compress.utils.atomic_io import atomic_torch_save, write_manifest_last
    manifest_path = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    # Drop any stale manifest from a prior interrupted run before the new
    # write so resumers never see an old manifest "approve" the new
    # partial .pt during the window between atomic_torch_save and
    # write_manifest_last.
    try:
        manifest_path.unlink(missing_ok=True)
    except OSError:
        pass
    atomic_torch_save(payload, out_path)
    write_manifest_last(
        out_path,
        manifest_path,
        schema_version=1,
        extra_meta={
            "artifact": "stage5_teacher_logits",
            "model": args.model,
            "num_samples": int(spec.num_sequences),
            "sequence_length": int(spec.sequence_length),
            "batch_size": batch_size,
            "calibration_seed_offset": 5,
        },
        # SHA-256 of a 30 GB file is ~3-5 min on NVMe; we compute once at
        # write time and store, so opt-in deep validation by operators is
        # cheap; default read path uses size + schema only.
        compute_sha256=True,
    )
    LOG.info("Saved (%.2f GB on disk) — manifest %s",
             out_path.stat().st_size / 1e9, manifest_path)

    # Optional Hub upload so a separate Stage 5 job (different bucket
    # mount) can fetch the cache directly.
    if args.upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload_repo, repo_type="model", private=True, exist_ok=True)
        LOG.info("Uploading → https://huggingface.co/%s", args.upload_repo)
        # Pattern O: manifest-LAST. Upload the payload .pt FIRST and wait
        # for the commit to return, then upload the MANIFEST.json. A
        # partial upload that fails between the two leaves the .pt on Hub
        # without a manifest → Stage 5's F-RK-1 reader fails loudly
        # (manifest missing) rather than mmap-loading a half-uploaded
        # payload and silently degrading KD.
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_path.name,
            repo_id=args.upload_repo,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=manifest_path.name,
            repo_id=args.upload_repo,
            repo_type="model",
        )
        LOG.info("Upload complete (payload + manifest)")

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Stage 5 teacher router logits")
    p.add_argument("--config", default="/mnt/cache/code/configs/qwen36_35b_a3b_30pct.yaml")
    p.add_argument("--code-root", default="/mnt/cache/code")
    p.add_argument("--model", default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.6-35B-A3B"))
    p.add_argument("--artifacts-dir", default="/mnt/cache/artifacts")
    p.add_argument("--output", default="/mnt/cache/artifacts/_stage5_teacher_logits.pt")
    p.add_argument(
        "--upload-repo",
        default=os.environ.get("TEACHER_LOGITS_REPO", "pirola/qwen3-6-35b-a3b-teacher-logits"),
        help="Hub model repo to upload the cache to. Empty string = skip upload.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(_main())
