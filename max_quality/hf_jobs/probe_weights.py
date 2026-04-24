"""Zero-GPU probe: read the cached safetensors index and print every key
matching `experts.*` / `shared_expert.*` with its tensor shape.

Runs on cpu-basic. Uses the already-cached HF snapshot in the bucket, so
no network + no model load.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.26.0",
#     "safetensors>=0.4.5",
#     "numpy>=1.26.0",
# ]
# ///

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    os.environ["HF_HOME"] = "/mnt/cache/hf_cache"
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen3.6-35B-A3B"
    root = Path(snapshot_download(model_id, cache_dir="/mnt/cache/hf_cache/hub"))
    print("snapshot at:", root)

    idx_file = root / "model.safetensors.index.json"
    index = json.loads(idx_file.read_text())
    weight_map = index["weight_map"]         # key -> shard filename
    print(f"Total tensor keys: {len(weight_map)}")

    # We need shapes, not just key names. safetensors stores shape in the
    # shard header — read one entry per unique shard to get all keys + shapes.
    from safetensors import safe_open
    shards = sorted({v for v in weight_map.values()})
    key_to_shape: dict[str, list[int]] = {}
    for shard in shards:
        with safe_open(root / shard, framework="numpy") as f:
            for k in f.keys():
                key_to_shape[k] = list(f.get_slice(k).get_shape())
    print(f"Total tensor entries across shards: {len(key_to_shape)}")

    # Focus: layer 0 MoE block.
    patterns = [
        r"\.layers\.0\.mlp\.experts\.",
        r"\.layers\.0\.mlp\.gate",
        r"\.layers\.0\.mlp\.shared_expert\.",
        r"\.layers\.0\.mlp\.shared_expert_gate",
    ]
    for pat in patterns:
        print(f"\n--- keys matching  {pat}  ---")
        for k in sorted(key_to_shape):
            if re.search(pat, k):
                print(f"  {k:80s}  shape={key_to_shape[k]}")

    # Cross-layer consistency: count how many layers have `mlp.experts.*`
    # and print the distinct key suffixes.
    suffixes = defaultdict(int)
    for k in key_to_shape:
        m = re.search(r"\.layers\.\d+\.mlp\.experts\.(.+)$", k)
        if m:
            suffixes[m.group(1)] += 1
    print(f"\n--- distinct experts.* suffixes across all 40 layers ---")
    for suf, count in sorted(suffixes.items()):
        print(f"  experts.{suf:60s}  present in {count} layers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
