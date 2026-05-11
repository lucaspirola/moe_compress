"""Local model-structure validator for Zyphra/ZAYA1-reasoning-base.

Builds the model on the meta device using the patched local transformers
fork at /home/lucas/ai/transformers-zaya1, then diffs the constructed
parameter names + shapes against the checkpoint's safetensors metadata.

No model weights are downloaded — only:
  - config.json  (~few KB)
  - safetensors headers (~few MB per shard, fetched via HTTP range)

So this script runs in seconds, uses <100 MB RAM, and iterates as fast as
you can edit /home/lucas/ai/transformers-zaya1.

Usage:
    /home/lucas/ai/moe_compress/knowledge_distillation_recovery/kdr/.venv-kdr/bin/python \\
        /home/lucas/ai/zaya1-load-test/inspect_shapes.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

import torch
from accelerate import init_empty_weights
from huggingface_hub import HfFileSystem
from transformers import AutoConfig, AutoModelForCausalLM

# Override either repo (HF Hub) or path (local directory) via argv or env:
#   inspect_shapes.py /mnt/d/models/Zyphra/ZAYA1-8B
#   MODEL=Zyphra/ZAYA1-8B inspect_shapes.py
DEFAULT_REPO = "Zyphra/ZAYA1-reasoning-base"
SOURCE = (
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("MODEL", DEFAULT_REPO)
)
IS_LOCAL = Path(SOURCE).is_dir()


def _fetch_safetensors_shape_map(source: str) -> dict[str, tuple[int, ...]]:
    """Pull every shard's safetensors header (just the JSON, no weights).

    Works for both local directories and HF Hub repo IDs. For local dirs we
    open the file directly; for repos we use HfFileSystem (HTTP range fetch).

    Returns a {param_name: shape_tuple} dict spanning all shards.
    """
    is_local = Path(source).is_dir()
    if is_local:
        # Prefer the safetensors index if present; otherwise glob shards directly.
        # Partial downloads sometimes lack model.safetensors.index.json — for
        # shape validation we don't need the param→shard mapping, just every
        # shard's header.
        index_path = Path(source) / "model.safetensors.index.json"
        if index_path.exists():
            shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            shards = sorted(p.name for p in Path(source).glob("model-*.safetensors"))
            single = Path(source) / "model.safetensors"
            if single.exists() and not shards:
                shards = [single.name]
            if not shards:
                raise FileNotFoundError(f"No safetensors files found under {source}")
    else:
        index_url = f"https://huggingface.co/{source}/raw/main/model.safetensors.index.json"
        with urlopen(index_url) as r:
            shards = sorted(set(json.load(r)["weight_map"].values()))
    shapes: dict[str, tuple[int, ...]] = {}
    if is_local:
        for shard in shards:
            shard_path = Path(source) / shard
            with shard_path.open("rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(header_len).decode())
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                shapes[k] = tuple(v["shape"])
            print(f"  shard {shard}: {len(header) - ('__metadata__' in header)} params")
    else:
        fs = HfFileSystem()
        for shard in shards:
            path = f"{source}/{shard}"
            with fs.open(path, "rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(header_len).decode())
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                shapes[k] = tuple(v["shape"])
            print(f"  shard {shard}: {len(header) - ('__metadata__' in header)} params")
    return shapes


def main() -> int:
    print(f"# Loading config from {SOURCE!r} ({'local' if IS_LOCAL else 'hf hub'})")
    config = AutoConfig.from_pretrained(SOURCE, trust_remote_code=True)
    print(
        f"  model_type={config.model_type}  hidden_size={config.hidden_size}  "
        f"num_hidden_layers={config.num_hidden_layers}  vocab={config.vocab_size}"
    )

    print(f"\n# Reading safetensors headers ({'local files' if IS_LOCAL else 'HTTP range'})")
    ckpt_shapes = _fetch_safetensors_shape_map(SOURCE)
    print(f"  total checkpoint params: {len(ckpt_shapes)}")

    print("\n# Instantiating model on meta device (using patched local fork)")
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.bfloat16
            )
    except Exception as e:
        print(f"  INSTANTIATION FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2

    expected_shapes = {n: tuple(p.shape) for n, p in model.state_dict().items()}
    print(f"  total model params: {len(expected_shapes)}")

    print("\n# === Diff ===")
    ckpt_names = set(ckpt_shapes)
    model_names = set(expected_shapes)
    only_in_ckpt = sorted(ckpt_names - model_names)
    only_in_model = sorted(model_names - ckpt_names)
    common = sorted(ckpt_names & model_names)

    print(f"common: {len(common)}")
    print(f"only in checkpoint (model doesn't build them): {len(only_in_ckpt)}")
    for n in only_in_ckpt[:20]:
        print(f"  {n}  shape={ckpt_shapes[n]}")
    if len(only_in_ckpt) > 20:
        print(f"  ... {len(only_in_ckpt) - 20} more")

    print(f"only in model (checkpoint missing them): {len(only_in_model)}")
    for n in only_in_model[:20]:
        print(f"  {n}  shape={expected_shapes[n]}")
    if len(only_in_model) > 20:
        print(f"  ... {len(only_in_model) - 20} more")

    shape_mismatches = [
        (n, ckpt_shapes[n], expected_shapes[n])
        for n in common
        if ckpt_shapes[n] != expected_shapes[n]
    ]
    print(f"shape mismatches (same name, different shape): {len(shape_mismatches)}")
    for n, ckpt, exp in shape_mismatches[:30]:
        print(f"  {n}  ckpt={ckpt}  model={exp}")
    if len(shape_mismatches) > 30:
        print(f"  ... {len(shape_mismatches) - 30} more")

    if not only_in_ckpt and not only_in_model and not shape_mismatches:
        print("\nALL PARAMS MATCH ✓ — the patched fork lines up with the checkpoint.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
