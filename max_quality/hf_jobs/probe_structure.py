"""Module-tree probe for Qwen/Qwen3.6-35B-A3B.

Loads the model (reusing the bucket-cached snapshot, so ~60s) and prints
the top-level module tree plus any submodule containing ``experts`` or
``gate``. Used to fix ``iter_moe_layers`` / ``_find_text_tower`` for
multimodal MoE layouts.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.5.0,<2.11.0",
#     "transformers>=4.57.0",
#     "accelerate>=1.0.0",
#     "huggingface_hub>=0.26.0",
#     "safetensors>=0.4.5",
#     "tokenizers>=0.20.0",
#     "einops>=0.8.0",
#     "sentencepiece>=0.2.0",
# ]
# ///

import os
import sys
from pathlib import Path


def main() -> int:
    os.environ["HF_HOME"] = "/mnt/cache/hf_cache"
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM

    model_id = "Qwen/Qwen3.6-35B-A3B"
    cfg = AutoConfig.from_pretrained(model_id)
    print(f"architectures: {cfg.architectures}")

    # Prefer the image-text-to-text auto class; fall back to causal.
    for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = auto_cls.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print(f"Loaded with {auto_cls.__name__}")
            break
        except Exception as err:
            print(f"{auto_cls.__name__} failed: {err}")
    else:
        return 1

    print(f"type(model)        = {type(model).__name__}")
    print(f"top-level children:")
    for name, _ in model.named_children():
        print(f"  {name}")
    print()

    # Walk down the wrapping chain to find `.layers`.
    to_visit = [("model", model)]
    seen = set()
    found_layers = []
    while to_visit:
        path, mod = to_visit.pop(0)
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if hasattr(mod, "layers"):
            try:
                n = len(mod.layers)
            except TypeError:
                n = "?"
            print(f"FOUND .layers at {path} — {n} entries; class={type(mod).__name__}")
            found_layers.append((path, mod))
        for cname, c in mod.named_children():
            if cname in ("layers",):
                continue
            to_visit.append((f"{path}.{cname}", c))

    print()
    # For the first text-tower hit, inspect the first layer's structure.
    if found_layers:
        path, tower = found_layers[0]
        layer0 = tower.layers[0]
        print(f"--- layer 0 structure (path={path}.layers[0], class={type(layer0).__name__}) ---")
        for name, child in layer0.named_children():
            print(f"  {name}: {type(child).__name__}")
        # Specifically look for the MoE block under any name.
        print(f"\n--- layer 0 modules with 'expert' or 'gate' in the name ---")
        for full_name, child in layer0.named_modules():
            if any(s in full_name.lower() for s in ("expert", "gate", "router", "moe")):
                print(f"  {full_name}: {type(child).__name__}")

        # Print the actual MoE block contents if found
        for full_name, child in layer0.named_modules():
            if hasattr(child, "experts"):
                print(f"\n--- MoE block at {full_name} ---")
                for n, c in child.named_children():
                    extra = ""
                    if hasattr(c, "__len__"):
                        try:
                            extra = f" (len={len(c)})"
                        except TypeError:
                            pass
                    print(f"  {n}: {type(c).__name__}{extra}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
